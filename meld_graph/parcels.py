"""
Helpers for Desikan-Killiany (DK) parcel vectors used by MoNetUnetParcelFusion.

Parcel vectors are stored either as one .npy file per subject (file_pattern "{subject_id}.npy"):

    <parcel_dir>/<subject_id>.npy   shape (2, n_parcels) or (2, n_parcels, parcel_dim)
                                    axis 0 = hemisphere, order [lh, rh]

or as one file per hemisphere (file_pattern e.g. "{subject_id}_{hemi}.npy", {hemi} = LH / RH):

    <parcel_dir>/<subject_id>_LH.npy, <subject_id>_RH.npy   shape (n_parcels,) or (n_parcels, parcel_dim)

In both cases the parcel axis follows the colour-table order of fsaverage_sym/label/lh.aparc.annot
(position 0 = 'unknown' ... 35 = 'insula' for the standard DK annot).

Each vertex of the level-7 mesh (fsaverage_sym, 163842 vertices) gets the vector of the parcel it
belongs to. Vertices without a parcel get a zero vector. Because the icospheres are nested (the first
N vertices of level L+1 are the vertices of level L), the vertex->parcel map of a coarser level L is
simply the first N_L entries of the level-7 map.

Parcel position = position of the parcel in the annot file's colour table (0, 1, 2, ...), whatever
structure numbers the colour table uses. (Some aparc.annot files number their entries like
FreeSurferColorLUT, e.g. 1000-1035 for ctx-lh-*; nibabel.freesurfer.read_annot then returns those
numbers instead of 0-35. read_annot_parcels below always returns positions 0..n-1.)
"""

import glob
import os
import logging

import numpy as np

from meld_graph.paths import MELD_PARAMS_PATH, DK_ATLAS_FILE, NVERT, BASE_PATH

log = logging.getLogger(__name__)


def read_annot_parcels(annot_file):
    """
    Read a FreeSurfer .annot file (old or new colour-table format).

    Returns:
        positions (np.ndarray): int64, shape (n_vertices,). For every vertex, the position (0..n-1) of its
            parcel in the colour table, in file order; -1 if the vertex has no parcel (annotation value 0
            or a value that is not in the colour table).
        names (list of str): parcel names in colour-table (file) order.
        table_ids (list of int): the structure number each colour-table entry has in the file
            (e.g. 0..35, or 1000..1035), for information.
    """
    with open(annot_file, "rb") as f:
        buf = f.read()
    pos = 0

    def ints(n=1):
        nonlocal pos
        out = np.frombuffer(buf, dtype=">i4", count=n, offset=pos).astype(np.int64)
        pos += 4 * n
        return out

    def string(n):
        nonlocal pos
        s = buf[pos:pos + n]
        pos += n
        return s.split(b"\x00")[0].decode("latin-1")

    n_vertices = int(ints()[0])
    vertex_data = ints(2 * n_vertices).reshape(n_vertices, 2)
    values = np.zeros(n_vertices, dtype=np.int64)
    values[vertex_data[:, 0]] = vertex_data[:, 1]

    if int(ints()[0]) == 0:
        raise ValueError(f"{annot_file} has no colour table")
    n = int(ints()[0])
    entries = []  # (table id, name, packed rgb)
    if n > 0:  # old format: n entries, numbered by position
        string(int(ints()[0]))  # original colour table path
        for i in range(n):
            name = string(int(ints()[0]))
            r, g, b, _ = ints(4)
            entries.append((i, name, int(r + g * 256 + b * 65536)))
    else:  # new format, version -n
        if -n != 2:
            raise ValueError(f"{annot_file}: unsupported annot colour-table version {-n}")
        ints()  # max structure number + 1
        string(int(ints()[0]))  # original colour table path
        for _ in range(int(ints()[0])):
            table_id = int(ints()[0])
            name = string(int(ints()[0]))
            r, g, b, _ = ints(4)
            entries.append((table_id, name, int(r + g * 256 + b * 65536)))

    colour_to_position = {}
    for position, (_, name, colour) in enumerate(entries):
        if colour in colour_to_position:
            log.warning(f"{annot_file}: parcels '{entries[colour_to_position[colour]][1]}' and '{name}' "
                        "have the same colour; vertices are assigned to the first")
            continue
        colour_to_position[colour] = position
    positions = np.full(n_vertices, -1, dtype=np.int64)
    for colour, position in colour_to_position.items():
        if colour != 0:
            positions[values == colour] = position
    unknown_values = np.unique(values[(positions < 0) & (values != 0)])
    if len(unknown_values):
        log.warning(f"{annot_file}: {len(unknown_values)} annotation values are not in the colour table; "
                    "those vertices get no parcel")
    return positions, [e[1] for e in entries], [e[0] for e in entries]


def load_dk_vertex_labels(n_parcels, annot_file=None):
    """
    Read the DK annotation and return the vertex -> parcel position map at level 7.

    Args:
        n_parcels (int): expected number of parcels (entries in the annot colour table).
        annot_file (str, optional): path to the .annot file. Default: MELD_PARAMS_PATH/fsaverage_sym/label/lh.aparc.annot

    Returns:
        labels (np.ndarray): int64, shape (NVERT,). Values 0..n_parcels-1 are parcel positions
            (colour-table order); vertices without a parcel are set to n_parcels, which the
            model maps to a zero vector.
    """
    if annot_file is None:
        annot_file = os.path.join(MELD_PARAMS_PATH, DK_ATLAS_FILE)
    if not os.path.isfile(annot_file):
        raise FileNotFoundError(f"DK annotation not found: {annot_file}")
    labels, names, _ = read_annot_parcels(annot_file)
    if len(labels) != NVERT:
        raise ValueError(f"{annot_file} has {len(labels)} vertices, expected {NVERT} (fsaverage_sym)")
    if len(names) != n_parcels:
        raise ValueError(
            f"parcel_fusion.n_parcels={n_parcels} but {annot_file} has {len(names)} parcels "
            f"in its colour table: {names}"
        )
    unassigned = labels < 0
    if unassigned.any():
        log.info(f"{unassigned.sum()} vertices have no DK parcel; they get a zero parcel vector")
    labels[unassigned] = n_parcels
    if labels.min() < 0 or labels.max() > n_parcels:
        raise ValueError(f"{annot_file}: parcel positions outside 0..{n_parcels}")
    return labels


def parcel_file_paths(subject_id, parcel_params):
    """
    Paths of a subject's parcel vector file(s), from parcel_params["file_pattern"]:
        "{subject_id}.npy"         one file per subject, shape (2, n_parcels[, D]), order lh, rh (default)
        "{subject_id}_{hemi}.npy"  one file per hemisphere ({hemi} = LH / RH), shape (n_parcels,) or (n_parcels, D)
    Returns a list with one path, or [LH path, RH path].
    """
    parcel_dir = parcel_params["parcel_dir"]
    if not os.path.isabs(parcel_dir):
        parcel_dir = os.path.join(BASE_PATH, parcel_dir)
    pattern = parcel_params.get("file_pattern", "{subject_id}.npy")
    if "{hemi}" in pattern:
        return [os.path.join(parcel_dir, pattern.format(subject_id=subject_id, hemi=hemi)) for hemi in ("LH", "RH")]
    return [os.path.join(parcel_dir, pattern.format(subject_id=subject_id))]


def detect_parcel_dim(parcel_dir, file_pattern="{subject_id}.npy"):
    """Numbers per parcel (D), read from the first file in parcel_dir that matches file_pattern."""
    per_hemisphere = "{hemi}" in file_pattern
    glob_pattern = file_pattern.format(subject_id="*", hemi="LH") if per_hemisphere else file_pattern.format(subject_id="*")
    files = sorted(glob.glob(os.path.join(parcel_dir, glob_pattern)))
    if not files:
        raise FileNotFoundError(f"No parcel vector files matching {glob_pattern} in {parcel_dir}")
    shape = np.load(files[0], mmap_mode="r").shape
    base_ndim = 1 if per_hemisphere else 2  # (n_parcels,) per hemisphere, or (2, n_parcels) per subject
    if len(shape) == base_ndim:
        return 1
    if len(shape) == base_ndim + 1:
        return int(shape[-1])
    raise ValueError(
        f"{files[0]} has shape {shape}; with file_pattern '{file_pattern}' expected "
        + ("(n_parcels,) or (n_parcels, D)" if per_hemisphere else "(2, n_parcels) or (2, n_parcels, D)")
    )


def load_parcel_vectors(subject_id, parcel_params):
    """
    Load one subject's parcel vectors.

    Args:
        subject_id (str): subject id, used in parcel_params["file_pattern"] (see parcel_file_paths)
        parcel_params (dict): data_parameters["parcel_fusion"]. Uses keys
            parcel_dir (str): absolute path, or relative to BASE_PATH
            file_pattern (str): "{subject_id}.npy" (default) or e.g. "{subject_id}_{hemi}.npy"
            n_parcels (int)
            parcel_dim (int)
            missing_subject (str): "error" (default) or "zeros"

    Returns:
        np.ndarray float32, shape (2, n_parcels, parcel_dim), order [lh, rh].
    """
    n_parcels = parcel_params["n_parcels"]
    parcel_dim = parcel_params["parcel_dim"]
    paths = parcel_file_paths(subject_id, parcel_params)

    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        if parcel_params.get("missing_subject", "error") == "zeros":
            log.warning(f"No parcel vector for {subject_id} ({missing}); using zeros")
            return np.zeros((2, n_parcels, parcel_dim), dtype=np.float32)
        raise FileNotFoundError(
            f"No parcel vector for {subject_id}: {missing}. Create missing healthy-hemisphere files with "
            "scripts/data_preparation/create_missing_parcel_vectors.py, or set "
            "parcel_fusion.missing_subject='zeros' to use zero vectors instead."
        )

    if len(paths) == 2:
        # one file per hemisphere: (n_parcels,) or (n_parcels, D) each -> (2, n_parcels[, D]), lh first
        hemis = [np.load(p) for p in paths]
        if hemis[0].shape != hemis[1].shape:
            raise ValueError(f"{paths[0]} has shape {hemis[0].shape} but {paths[1]} has shape {hemis[1].shape}")
        vec = np.stack(hemis)
    else:
        vec = np.load(paths[0])
    if vec.ndim == 2:
        vec = vec[:, :, None]
    expected = (2, n_parcels, parcel_dim)
    if vec.shape != expected:
        raise ValueError(f"{paths} give shape {vec.shape}, expected {expected} (one file per hemisphere: "
                         f"{(n_parcels,) if parcel_dim == 1 else (n_parcels, parcel_dim)} each)")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{paths} contain NaN or inf values")
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
