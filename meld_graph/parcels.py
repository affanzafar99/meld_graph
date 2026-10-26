"""
Helpers for Desikan-Killiany (DK) parcel vectors used by MoNetUnetParcelFusion.

Per-subject parcel vectors are stored as one .npy file per subject:

    <parcel_dir>/<subject_id>.npy   shape (2, n_parcels) or (2, n_parcels, parcel_dim)
                                    axis 0 = hemisphere, order [lh, rh]
                                    axis 1 = parcel, in FreeSurfer aparc order, i.e. the
                                             colour-table order of fsaverage_sym/label/lh.aparc.annot
                                             (index 0 = 'unknown' ... 35 = 'insula' for the standard DK annot)

Each vertex of the level-7 mesh (fsaverage_sym, 163842 vertices) gets the vector of the parcel it
belongs to. Vertices without a parcel (annot label -1) get a zero vector. Because the icospheres are
nested (the first N vertices of level L+1 are the vertices of level L), the vertex->parcel map of a
coarser level L is simply the first N_L entries of the level-7 map.
"""

import os
import logging

import numpy as np
import nibabel as nb

from meld_graph.paths import MELD_PARAMS_PATH, DK_ATLAS_FILE, NVERT, BASE_PATH

log = logging.getLogger(__name__)


def load_dk_vertex_labels(n_parcels, annot_file=None):
    """
    Read the DK annotation and return the vertex -> parcel index map at level 7.

    Args:
        n_parcels (int): expected number of parcels (rows of the annot colour table).
        annot_file (str, optional): path to the .annot file. Default: MELD_PARAMS_PATH/fsaverage_sym/label/lh.aparc.annot

    Returns:
        labels (np.ndarray): int64, shape (NVERT,). Values 0..n_parcels-1 are parcel indices
            (colour-table order); vertices without a parcel are set to n_parcels, which the
            model maps to a zero vector.
    """
    if annot_file is None:
        annot_file = os.path.join(MELD_PARAMS_PATH, DK_ATLAS_FILE)
    if not os.path.isfile(annot_file):
        raise FileNotFoundError(f"DK annotation not found: {annot_file}")
    labels, ctab, names = nb.freesurfer.read_annot(annot_file)
    if len(labels) != NVERT:
        raise ValueError(f"{annot_file} has {len(labels)} vertices, expected {NVERT} (fsaverage_sym)")
    if len(names) != n_parcels:
        names_str = [n.decode() if isinstance(n, bytes) else n for n in names]
        raise ValueError(
            f"parcel_fusion.n_parcels={n_parcels} but {annot_file} has {len(names)} parcels "
            f"in its colour table: {names_str}"
        )
    labels = labels.astype(np.int64)
    unassigned = labels < 0
    if unassigned.any():
        log.info(f"{unassigned.sum()} vertices have no DK parcel; they get a zero parcel vector")
    labels[unassigned] = n_parcels
    return labels


def load_parcel_vectors(subject_id, parcel_params):
    """
    Load one subject's parcel vectors.

    Args:
        subject_id (str): subject id; the file is <parcel_dir>/<subject_id>.npy
        parcel_params (dict): data_parameters["parcel_fusion"]. Uses keys
            parcel_dir (str): absolute path, or relative to BASE_PATH
            n_parcels (int)
            parcel_dim (int)
            missing_subject (str): "error" (default) or "zeros"

    Returns:
        np.ndarray float32, shape (2, n_parcels, parcel_dim), order [lh, rh].
    """
    n_parcels = parcel_params["n_parcels"]
    parcel_dim = parcel_params["parcel_dim"]
    parcel_dir = parcel_params["parcel_dir"]
    if not os.path.isabs(parcel_dir):
        parcel_dir = os.path.join(BASE_PATH, parcel_dir)
    path = os.path.join(parcel_dir, f"{subject_id}.npy")

    if not os.path.isfile(path):
        if parcel_params.get("missing_subject", "error") == "zeros":
            log.warning(f"No parcel vector for {subject_id} ({path}); using zeros")
            return np.zeros((2, n_parcels, parcel_dim), dtype=np.float32)
        raise FileNotFoundError(
            f"No parcel vector for {subject_id}: {path}. "
            "Set parcel_fusion.missing_subject='zeros' to use zero vectors instead."
        )

    vec = np.load(path)
    if vec.ndim == 2:
        vec = vec[:, :, None]
    expected = (2, n_parcels, parcel_dim)
    if vec.shape != expected:
        raise ValueError(f"{path} has shape {vec.shape}, expected {expected} (or {expected[:2]} if parcel_dim=1)")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{path} contains NaN or inf values")
    return vec.astype(np.float32)


def map_parcels_to_vertices(parcel_vec, vertex_labels):
    """
    Numpy reference for the mapping done inside the model.

    Args:
        parcel_vec: (n_parcels, D) vector for one hemisphere
        vertex_labels: (n_vertices,) parcel index per vertex, n_parcels = no parcel

    Returns:
        (n_vertices, D) array: row v is parcel_vec[vertex_labels[v]], or zeros if no parcel.
    """
    padded = np.concatenate([parcel_vec, np.zeros((1, parcel_vec.shape[1]), dtype=parcel_vec.dtype)], axis=0)
    return padded[vertex_labels]
