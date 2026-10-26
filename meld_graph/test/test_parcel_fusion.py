"""
Tests for MoNetUnetParcelFusion, encoder freezing and pretrained-weight loading.

Run from the repository root (icospheres are loaded from data/icospheres):
    pytest meld_graph/test/test_parcel_fusion.py -v

Uses small layer sizes so it runs on CPU in about a minute; no MELD data needed.
"""

import copy
import logging
import types

import numpy as np
import pytest
import torch
import torch_geometric

from meld_graph.models import MoNetUnet, MoNetUnetParcelFusion, get_model_inputs
from meld_graph.parcels import map_parcels_to_vertices
from meld_graph.experiment import Experiment
from meld_graph.augment import Augment

N_FEAT = 3
N_PARCELS = 36
PARCEL_DIM = 2
LAYER_SIZES = [[4, 4], [4, 4], [4, 4], [4, 4], [8, 8], [8, 8], [8, 8]]
COMMON = dict(
    num_features=N_FEAT,
    layer_sizes=LAYER_SIZES,
    icosphere_params={"distance_type": "exact", "conv_type": "SpiralConv"},
    conv_type="SpiralConv",
    spiral_len=7,
    deep_supervision=[2, 3, 4, 5, 6],
    activation_fn="leaky_relu",
    classification_head=True,
)


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    base = MoNetUnet(**COMMON)
    fused = MoNetUnetParcelFusion(**COMMON, parcel_dim=PARCEL_DIM, n_parcels=N_PARCELS)
    result = fused.load_state_dict(base.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert all(k.startswith("parcel_fusion.") for k in result.missing_keys)
    base.to(torch.device("cpu"))
    fused.to(torch.device("cpu"))
    return base, fused


def _sample(seed, n_vertices):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(n_vertices, N_FEAT)), dtype=torch.float32)
    labels = rng.integers(0, N_PARCELS + 1, size=n_vertices)  # N_PARCELS = "no parcel"
    vec = rng.normal(size=(N_PARCELS, PARCEL_DIM)).astype(np.float32)
    return x, torch.tensor(labels, dtype=torch.int64), torch.tensor(vec)


def _randomise_fusion(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parcel_fusion.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)


def test_fusion_modules_created(models):
    _, fused = models
    assert fused.decoder_levels == [2, 3, 4, 5, 6, 7]
    assert sorted(fused.parcel_fusion.keys(), key=int) == ["2", "3", "4", "5", "6", "7"]
    # channels match the decoder block outputs: level 7 -> layer_sizes[0][0], level 2 -> layer_sizes[5][0]
    assert fused.parcel_fusion["7"].channels == LAYER_SIZES[0][0]
    assert fused.parcel_fusion["2"].channels == LAYER_SIZES[5][0]


def test_identity_at_init_matches_pretrained(models):
    """With scale = 0 the fused model reproduces the base model exactly (single hemisphere)."""
    base, fused = models
    fused = copy.deepcopy(fused)
    with torch.no_grad():
        for m in fused.parcel_fusion.values():
            m.scale.zero_()
    x, labels, vec = _sample(0, base.n_vertices)
    with torch.no_grad():
        out_base = base(x)
        out_fused = fused(x, parcel_vec=vec, parcel_labels=labels)
    for key in out_base:
        torch.testing.assert_close(out_fused[key], out_base[key], rtol=0, atol=0)


def test_parcel_vector_changes_output(models):
    _, fused = models
    fused = copy.deepcopy(fused)
    _randomise_fusion(fused)
    x, labels, vec = _sample(0, fused.n_vertices)
    with torch.no_grad():
        out1 = fused(x, parcel_vec=vec, parcel_labels=labels)["log_softmax"]
        out2 = fused(x, parcel_vec=vec + 1.0, parcel_labels=labels)["log_softmax"]
    assert not torch.allclose(out1, out2)


def test_batch_matches_single_samples_and_pyg_batching(models):
    """Each hemisphere in a batch is decoded with its own skips and its own parcel vector."""
    _, fused = models
    fused = copy.deepcopy(fused)
    _randomise_fusion(fused)
    data_list = []
    for seed in (0, 1):
        x, labels, vec = _sample(seed, fused.n_vertices)
        data_list.append(torch_geometric.data.Data(x=x, parcel_vec=vec, parcel_labels=labels, num_nodes=len(x)))
    batch = torch_geometric.data.Batch.from_data_list(data_list)
    assert batch.parcel_vec.shape == (2 * N_PARCELS, PARCEL_DIM)
    assert batch.parcel_labels.shape == (2 * fused.n_vertices,)
    # labels must not be shifted by PyG batching
    assert torch.equal(batch.parcel_labels[fused.n_vertices:], data_list[1].parcel_labels)
    with torch.no_grad():
        out_batch = fused(batch.x, **get_model_inputs(batch))["log_softmax"]
        singles = [fused(d.x, **get_model_inputs(d))["log_softmax"] for d in data_list]
    torch.testing.assert_close(out_batch, torch.cat(singles), rtol=1e-5, atol=1e-6)


def test_parcel_mapping_per_level(models):
    _, fused = models
    _, labels, vec = _sample(3, fused.n_vertices)
    padded = torch.cat([vec, vec.new_zeros(1, PARCEL_DIM)])
    sample_inputs = {"parcel_vec": padded, "parcel_labels": labels}
    for level in fused.decoder_levels:
        n = fused.n_vertices_per_level[level]
        got = fused.parcel_features_at_level(sample_inputs, level).numpy()
        expected = map_parcels_to_vertices(vec.numpy(), labels[:n].numpy())
        np.testing.assert_array_equal(got, expected)
        # vertices without a parcel get zeros
        assert np.all(got[labels[:n].numpy() == N_PARCELS] == 0)


def test_freeze_encoder(models):
    _, fused = models
    fused = copy.deepcopy(fused)
    _randomise_fusion(fused)
    fused.freeze_encoder()
    x, labels, vec = _sample(0, fused.n_vertices)
    out = fused(x, parcel_vec=vec, parcel_labels=labels)
    loss = out["log_softmax"][:, 1].mean() + out["hemi_log_softmax"].sum()
    loss.backward()
    for name, p in fused.named_parameters():
        if name.startswith("encoder_conv_layers."):
            assert not p.requires_grad and p.grad is None, name
        elif name.startswith(("parcel_fusion.", "decoder_conv_layers.", "fc.", "hemi_classification_head.")):
            assert p.requires_grad and p.grad is not None, name


def test_missing_inputs_raise(models):
    base, fused = models
    x, labels, vec = _sample(0, fused.n_vertices)
    with pytest.raises(ValueError):
        fused(x)
    with pytest.raises(TypeError):
        base(x, parcel_vec=vec, parcel_labels=labels)


def test_load_pretrained_ensemble_member(models, tmp_path):
    base, fused = models
    members = [copy.deepcopy(base) for _ in range(2)]
    with torch.no_grad():
        for p in members[1].parameters():
            p.add_(1.0)
    ensemble_state = {}
    for k, m in enumerate(members):
        ensemble_state.update({f"models.{k}.{key}": val for key, val in m.state_dict().items()})
    path = tmp_path / "best_model.pt"
    torch.save(ensemble_state, path)

    target = MoNetUnetParcelFusion(**COMMON, parcel_dim=PARCEL_DIM, n_parcels=N_PARCELS)
    fake_exp = types.SimpleNamespace(model=target, log=logging.getLogger("test"))
    Experiment.load_pretrained_weights(fake_exp, str(path), ensemble_member=1)
    for key, val in members[1].state_dict().items():
        torch.testing.assert_close(target.state_dict()[key].cpu(), val.cpu())
    # parcel fusion layers keep their initialisation: scale is still 0
    assert all(torch.all(m.scale == 0) for m in target.parcel_fusion.values())

    with pytest.raises(ValueError):  # ensemble without choosing a member
        Experiment.load_pretrained_weights(fake_exp, str(path))

    # a checkpoint that does not match must fail loudly
    bad_state = {k.replace("encoder_conv_layers", "encoder_layers"): v for k, v in members[0].state_dict().items()}
    torch.save(bad_state, tmp_path / "bad.pt")
    with pytest.raises(RuntimeError):
        Experiment.load_pretrained_weights(fake_exp, str(tmp_path / "bad.pt"))


def test_augmentation_keeps_parcels_aligned():
    """Spinning/warping/flipping move parcel_labels together with the features."""
    rng = np.random.default_rng(0)
    n = 163842
    parcel_labels = rng.integers(0, N_PARCELS + 1, size=n)
    tdd = {
        "features": np.stack([parcel_labels.astype(np.float32), rng.normal(size=n)], axis=1),
        "labels": (rng.random(n) < 0.01).astype(int),
        "distances": rng.random(n).astype(np.float32),
        "parcel_labels": parcel_labels,
    }
    for name in ("spinning", "warping", "flipping"):
        indices = np.load(f"data/{name}/{name}_ico7_{3 if name == 'flipping' else 10}.npy")[0].astype(int)
        out = Augment.apply_indices(None, indices, dict(tdd))
        np.testing.assert_array_equal(out["features"][:, 0], out["parcel_labels"].astype(np.float32))
        assert not np.array_equal(out["parcel_labels"], parcel_labels)
