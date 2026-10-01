"""
CellWHISPER core routines.

- spatial neighbor graph (symmetric k-NN, optionally distance-capped)
- gene-pair-specific "whisper networks" (gap junction or ligand-receptor)
- analytical null (mean, variance) and z-score for cell-type x signaling-gene quadruplets

Entry point for most users: run_cellwhisper_pair_adata().
"""

import os
import time
import warnings

import numpy as np
from sklearn.neighbors import (
    kneighbors_graph,
    radius_neighbors_graph,
    NearestNeighbors,
)

import pandas as pd  # backward compatibility 
import networkx as nx  

from utils import pkl_load, pkl_save
import scipy.sparse as sp

warnings.filterwarnings("ignore")

MIN_VALUE = 1e-16  # numerical stabilizer


# ----------------------------------------------------------------------
# Helpers: annotation normalization (critical for non-iid correctness)
# ----------------------------------------------------------------------
def _normalize_annot_list(annot_list):
    """
    Normalize annotations to a 1D numpy array of scalar strings.

    Accepts:
      - 1D array-like of strings
      - (n,1) arrays (e.g. DataFrame.values) whose elements are 1-element arrays
      - pandas Series / DataFrame column

    Returns
    -------
    a : ndarray, shape (n,), dtype=str
    """
    a = np.asarray(annot_list).reshape(-1)

    # If elements are 1-element ndarrays (e.g., array(['Microglia'])), unwrap.
    if len(a) > 0 and isinstance(a[0], np.ndarray):
        a = np.array([x.item() if x.size == 1 else x[0] for x in a], dtype=object)

    # Cast to string for stable comparisons / list.index
    return a.astype(str)

def _cache_path(f_object, num_neighbor, max_distance):
    if f_object is None:
        return None
    root, ext = os.path.splitext(f_object)
    tag = f"_k{num_neighbor}"
    if max_distance is not None:
        tag += f"_d{max_distance:g}"
    return f"{root}{tag}{ext}"


# ----------------------------------------------------------------------
# Spatial neighbor graph construction
# ----------------------------------------------------------------------
def build_neighbor_network(df_spatial, num_neighbor):
    """
    Build a symmetric k-NN adjacency matrix (boolean).

    Parameters
    ----------
    df_spatial : array-like, shape (n_cells, n_dims)
        Spatial coordinates (e.g. x, y).
    num_neighbor : int
        Number of nearest neighbors (k).

    Returns
    -------
    graph_dist : ndarray of bool, shape (n_cells, n_cells)
        Symmetric adjacency matrix where True indicates an undirected
        edge between neighboring cells.
    """
    A = kneighbors_graph(
        df_spatial,
        num_neighbor,
        mode="connectivity",
        include_self=False,
    )
    graph_dist = np.array(A.toarray(), dtype=bool)
    graph_dist = np.multiply(graph_dist, graph_dist.T)
    return graph_dist


def build_neighbor_network_knn_n_dist(df_spatial, num_neighbor, max_distance=None):
    """
    Build a neighbor graph that is the intersection of:
      - k-NN graph (k = num_neighbor)
      - radius graph (within max_distance)

    Parameters
    ----------
    df_spatial : array-like, shape (n_cells, n_dims)
    num_neighbor : int
    max_distance : float or None

    Returns
    -------
    graph_bool : ndarray of bool, shape (n_cells, n_cells)
    """
    knn_graph = kneighbors_graph(
        df_spatial,
        num_neighbor,
        mode="connectivity",
        include_self=False,
    )
    knn_graph = knn_graph.minimum(knn_graph.T)

    if max_distance is None:
        nbrs = NearestNeighbors(n_neighbors=2).fit(df_spatial)
        distances, _ = nbrs.kneighbors(df_spatial)
        max_distance = np.mean(distances[:, 1])

    radius_graph = radius_neighbors_graph(
        df_spatial,
        radius=max_distance,
        mode="connectivity",
        include_self=False,
    )

    final_graph = knn_graph.multiply(radius_graph)
    graph_bool = final_graph.toarray().astype(bool)
    return graph_bool


# ----------------------------------------------------------------------
# Whisper network builder for a given signaling gene pair
# ----------------------------------------------------------------------
class BuildWhisperNetwork:
    """
    Container for spatial neighbor graph and gene-pair-specific whisper networks.
    """

    def __init__(self, df_spatial, annot_df, annot, num_neighbor=5, max_distance=None):
        if max_distance is None:
            self.graph_dist = build_neighbor_network(df_spatial, num_neighbor)
        else:
            self.graph_dist = build_neighbor_network_knn_n_dist(
                df_spatial, num_neighbor, max_distance
            )
        self.annot_df = annot_df
        self.annot = annot

    def create_whisper_network(self, df_cx1, df_cx2, cx_thresh1, cx_thresh2):
        graph_gene_pair = self.whisper_network(df_cx1, df_cx2, cx_thresh1, cx_thresh2)
        self.graph_whisper = np.multiply(self.graph_dist, graph_gene_pair)

    def whisper_network(self, g1_exp, g2_exp, cx_thresh1, cx_thresh2):
        g1_exp_bool = np.zeros(g1_exp.shape, dtype=bool).reshape(-1, 1)
        g2_exp_bool = np.zeros(g2_exp.shape, dtype=bool).reshape(-1, 1)

        g1_exp_bool[g1_exp > cx_thresh1] = True
        g2_exp_bool[g2_exp > cx_thresh2] = True

        return np.matmul(g1_exp_bool, g2_exp_bool.T)


def create_or_load_whisper_graph(df_spatial, annot_df, annot, f_gj_object,
                                 num_neighbor=5, max_distance=None):
    f_gj_object = _cache_path(f_gj_object, num_neighbor, max_distance)

    if f_gj_object is None:
        return BuildWhisperNetwork(df_spatial, annot_df, annot, num_neighbor, max_distance)

    if os.path.exists(f_gj_object):
        print(f"loading neighbor object: {f_gj_object}")
        return pkl_load(f_gj_object)

    gj = BuildWhisperNetwork(df_spatial, annot_df, annot, num_neighbor, max_distance)
    print(f"saving neighbor object: {f_gj_object}")
    pkl_save(gj, f_gj_object)
    return gj

def create_whisper_graph(df_spatial, annot_df, annot,
                         f_object=None,
                         num_neighbor=5, max_distance=None):
    return create_or_load_whisper_graph(df_spatial, annot_df, annot, f_object,
                                        num_neighbor, max_distance)


# ----------------------------------------------------------------------
# Analytical null model (mean and std of edge counts)
# ----------------------------------------------------------------------
def null_prob(g1_exp, g2_exp, cx_thresh1, cx_thresh2, cell_type_list, annot_list):
    annot_list = _normalize_annot_list(annot_list)

    g1_exp_bool = np.zeros(g1_exp.shape, dtype=bool).reshape(-1, 1)
    g2_exp_bool = np.zeros(g2_exp.shape, dtype=bool).reshape(-1, 1)

    g1_exp_bool[g1_exp > cx_thresh1] = True
    g2_exp_bool[g2_exp > cx_thresh2] = True

    p1_list = np.ones(len(cell_type_list)).reshape(-1, 1)
    p2_list = np.ones(len(cell_type_list)).reshape(-1, 1)

    cell_type_index_list = np.array([cell_type_list.index(i) for i in annot_list])

    for i in range(len(cell_type_list)):
        cells_index_ct_list = cell_type_index_list == i
        temp_cx1 = g1_exp_bool[cells_index_ct_list]
        temp_cx2 = g2_exp_bool[cells_index_ct_list]
        p1_list[i] = temp_cx1.sum() / len(temp_cx1)
        p2_list[i] = temp_cx2.sum() / len(temp_cx2)

    p1_p2_mat = np.matmul(p1_list, p2_list.T)
    return p1_p2_mat, p1_list, p2_list, g1_exp_bool, g2_exp_bool


def null_std(
    adj_mat,
    cell_type_list,
    annot_list,
    p1_list,
    p2_list,
    g1_exp_bool,
    g2_exp_bool,
    num_prox_ct_pair,
    mean_mat_curr,
):
    annot_list = _normalize_annot_list(annot_list)

    cell_type_index_list = np.array([cell_type_list.index(i) for i in annot_list])
    n_ct = len(cell_type_list)

    std_ct_pair = -100 * np.ones((n_ct, n_ct), dtype=np.float64)
    cells_index_ct_list = [None] * n_ct

    # Flatten expression flags
    g1_exp_bool = np.asarray(g1_exp_bool, dtype=bool).reshape(-1)
    g2_exp_bool = np.asarray(g2_exp_bool, dtype=bool).reshape(-1)

    # Promote numeric arrays
    num_prox_ct_pair = np.asarray(num_prox_ct_pair, dtype=np.float64)
    mean_mat_curr = np.asarray(mean_mat_curr, dtype=np.float64)
    p1_list = np.asarray(p1_list, dtype=np.float64)
    p2_list = np.asarray(p2_list, dtype=np.float64)

    for i in range(n_ct):
        cells_index_ct_list[i] = (cell_type_index_list == i)

    for i in range(n_ct):
        temp = adj_mat[cells_index_ct_list[i], :]

        # scalar p1
        p1 = float(p1_list[i]) if p1_list.ndim == 1 else float(p1_list[i, 0])

        for j in range(n_ct):
            adj_mat_ct = temp[:, cells_index_ct_list[j]]

            # scalar p2
            p2 = float(p2_list[j]) if p2_list.ndim == 1 else float(p2_list[j, 0])

            deg_ct = adj_mat_ct.sum(axis=1).astype(np.float64)
            # deg_ct[g1_exp_bool[cells_index_ct_list[i]] == 0] = 0.0
            deg_ct[deg_ct < 2] = 0.0
            # num_case1 = float((deg_ct * (deg_ct - 1) / 2).sum())
            num_case1 = (deg_ct * (deg_ct - 1)).sum()

            deg_ct = adj_mat_ct.sum(axis=0).astype(np.float64)
            # deg_ct[g2_exp_bool[cells_index_ct_list[j]] == 0] = 0.0
            deg_ct[deg_ct < 2] = 0.0
            # num_case2 = float((deg_ct * (deg_ct - 1) / 2).sum())
            num_case2 = (deg_ct * (deg_ct - 1)).sum()

            # nprox = float(num_prox_ct_pair[i, j])
            # nprox2 = nprox * nprox

            nprox2 = (num_prox_ct_pair[i][j] ** 2
                       - num_prox_ct_pair[i][j])

            if i != j:
                curr_var = (
                    num_case1 * p1 * (p2 ** 2)
                    + num_case2 * (p1 ** 2) * p2
                    + (nprox2 - num_case1 - num_case2) * (p1 ** 2) * (p2 ** 2)
                )
            else:
                curr_var = (
                    num_case1 * p1 * (p2 ** 2)
                    + (nprox2 - num_case1) * (p1 ** 2) * (p2 ** 2)
                )

            inside = curr_var + mean_mat_curr[i, j] - mean_mat_curr[i, j] ** 2
            if inside < 0 and inside > -1e-9:
                inside = 0.0

            std_ct_pair[i, j] = np.sqrt(inside)

    return std_ct_pair + MIN_VALUE


def cal_ct_pair_count(adj_mat, cell_type_list, annot_list):
    annot_list = _normalize_annot_list(annot_list)

    cell_type_index_list = np.array([cell_type_list.index(i) for i in annot_list])
    n_ct = len(cell_type_list)

    # preserve original behavior (dtype=int), but int64 is safer and does not change values
    ct_pair_count = np.ones((n_ct, n_ct), dtype=np.int64)

    cells_index_ct_list = [None] * n_ct
    for i in range(n_ct):
        cells_index_ct_list[i] = (cell_type_index_list == i)

    for i in range(n_ct):
        temp = adj_mat[cells_index_ct_list[i], :]
        for j in range(n_ct):
            adj_mat_ct = temp[:, cells_index_ct_list[j]]
            if i != j:
                ct_pair_count[i, j] = int(adj_mat_ct.sum())
            else:
                ct_pair_count[i, i] = int(np.triu(adj_mat_ct, 1).sum())

    return ct_pair_count


# ----------------------------------------------------------------------
# Wrapper: run CellWHISPER test for a single gene pair
# ----------------------------------------------------------------------
def run_cellwhisper_pair(
    df_spatial,
    df_cx1,
    df_cx2,
    annot_df,
    percentile=75,
    mode="non_iid",
    annot="annotation",
    f_gj_object=None,
    num_neighbor=5, 
    max_distance=None,
):
    gj = create_whisper_graph(df_spatial, annot_df, annot, f_gj_object, num_neighbor, max_distance)

    cx_thresh1 = np.percentile(df_cx1, percentile)
    cx_thresh2 = np.percentile(df_cx2, percentile)
    print(f"thresh g1 > {cx_thresh1} thresh 2 > {cx_thresh2}")

    gj.create_whisper_network(df_cx1, df_cx2, cx_thresh1, cx_thresh2)

    annot_vec = _normalize_annot_list(annot_df[annot].to_numpy() if hasattr(annot_df, "__getitem__") else annot_df)
    cell_type_list = list(np.unique(annot_vec))

    p1_p2_mat, p1_list, p2_list, g1_exp_bool, g2_exp_bool = null_prob(
        df_cx1,
        df_cx2,
        cx_thresh1,
        cx_thresh2,
        cell_type_list,
        annot_vec,
    )

    num_gj_ct_pair = cal_ct_pair_count(gj.graph_whisper, cell_type_list, annot_vec)
    num_prox_ct_pair = cal_ct_pair_count(gj.graph_dist, cell_type_list, annot_vec)

    mean_mat_curr = np.multiply(num_prox_ct_pair, p1_p2_mat)

    if mode == "iid":
        std_mat_curr = np.sqrt(np.multiply(mean_mat_curr, (1 - p1_p2_mat)) + MIN_VALUE)
    else:
        std_mat_curr = null_std(
            gj.graph_dist,
            cell_type_list,
            annot_vec,
            p1_list,
            p2_list,
            g1_exp_bool,
            g2_exp_bool,
            num_prox_ct_pair,
            mean_mat_curr,
        )

    z_score_gj_pair = (num_gj_ct_pair - mean_mat_curr) / (std_mat_curr + 1e-32)

    return (
        num_gj_ct_pair,
        num_prox_ct_pair,
        z_score_gj_pair,
        mean_mat_curr,
        std_mat_curr,
        gj.graph_whisper,  
    )


def run_cellwhisper_pair_adata(
    adata,
    gene1,
    gene2,
    annot="annotation",
    spatial_key="spatial",
    percentile=75,
    mode="non_iid",
    graph_cache=None,
    store_key=None,
    num_neighbor=5, 
    max_distance=None,
    return_graph=False
):
    df_spatial = adata.obsm[spatial_key]
    annot_df = adata.obs[annot].to_frame()

    df_cx1 = adata[:, gene1].X.toarray()
    df_cx2 = adata[:, gene2].X.toarray()

    (num_whisper_ct_pair, num_prox_ct_pair, z_score_ct_pair, mean_mat_curr, std_mat_curr, graph_whisper) = run_cellwhisper_pair(
        df_spatial=df_spatial,
        df_cx1=df_cx1,
        df_cx2=df_cx2,
        annot_df=annot_df,
        percentile=percentile,
        mode=mode,
        annot=annot,
        f_gj_object=graph_cache,
        num_neighbor=num_neighbor,       
        max_distance=max_distance, 
    )

    # Keep cell types consistent with the core call
    annot_vec = _normalize_annot_list(annot_df[annot].to_numpy())
    cell_type_list = list(np.unique(annot_vec))

    result = {
        "gene1": gene1,
        "gene2": gene2,
        "cell_types": cell_type_list,
        "num_whisper": num_whisper_ct_pair,
        "num_prox": num_prox_ct_pair,
        "z_score": z_score_ct_pair,
        "mean": mean_mat_curr,
        "std": std_mat_curr,
        "percentile": percentile,
        "mode": mode,
        "num_neighbor": num_neighbor,
        "max_distance": max_distance,
    }
    if return_graph:
        result["graph_whisper"] = sp.csr_matrix(graph_whisper)

    if store_key is not None:
        adata.uns[store_key] = result

    return result

