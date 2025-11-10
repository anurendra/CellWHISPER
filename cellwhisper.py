"""
CellWHISPER core routines

This module implements:
- Construction of spatial neighbor graphs
- Gene-pair-specific "whisper networks" (gap junction or ligand–receptor)
- Analytical null model (mean / variance) and z-score computation
for cell-type–signaling-gene quadruplets.

Behavior and numerical results are preserved.
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

import pandas as pd  # kept for backward compatibility even if unused
import networkx as nx  # kept for backward compatibility even if unused

from utils import pkl_load, pkl_save

warnings.filterwarnings("ignore")

MIN_VALUE = 1e-16  # numerical stabilizer


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
    # Convert to dense bool and symmetrize
    graph_dist = np.array(A.toarray(), dtype=bool)
    graph_dist = np.multiply(graph_dist, graph_dist.T)

    # print(f'total edges in graph with dist b/n neighbors {num_neighbor} is {np.triu(graph_dist,1).sum()}')

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
        Number of nearest neighbors for k-NN graph.
    max_distance : float or None
        Distance threshold for radius graph. If None, a default is
        estimated as the mean distance to each cell's 2nd nearest neighbor.

    Returns
    -------
    graph_bool : ndarray of bool, shape (n_cells, n_cells)
        Symmetric adjacency matrix indicating edges that satisfy both
        k-NN and distance constraints.
    """
    # k-NN graph (not necessarily symmetric)
    knn_graph = kneighbors_graph(
        df_spatial,
        num_neighbor,
        mode="connectivity",
        include_self=False,
    )
    # Symmetrize
    knn_graph = knn_graph.minimum(knn_graph.T)

    # If no radius is provided, estimate a characteristic scale
    if max_distance is None:
        nbrs = NearestNeighbors(n_neighbors=2).fit(df_spatial)
        distances, _ = nbrs.kneighbors(df_spatial)
        # distance to the 2nd neighbor for each point, then average
        max_distance = np.mean(distances[:, 1])

    radius_graph = radius_neighbors_graph(
        df_spatial,
        radius=max_distance,
        mode="connectivity",
        include_self=False,
    )

    # Intersection: must be both in k-NN and within max_distance
    final_graph = knn_graph.multiply(radius_graph)

    # Convert to boolean adjacency matrix
    graph_bool = final_graph.toarray().astype(bool)

    return graph_bool


# ----------------------------------------------------------------------
# Whisper network builder for a given signaling gene pair
# ----------------------------------------------------------------------
class BuildWhisperNetwork:
    """
    Container for spatial neighbor graph and gene-pair-specific whisper networks.

    Parameters
    ----------
    df_spatial : array-like, shape (n_cells, n_dims)
        Spatial coordinates.
    annot_df : pandas.DataFrame
        DataFrame with a single column containing cell-type annotations.
    annot : str
        Column name in annot_df that holds the annotation labels.
    num_neighbor : int, default 5
        k for k-NN graph.
    """

    def __init__(self, df_spatial, annot_df, annot, num_neighbor=5):
        self.graph_dist = build_neighbor_network(df_spatial, num_neighbor)
        self.annot_df = annot_df
        self.annot = annot

    def create_whisper_network(self, df_cx1, df_cx2, cx_thresh1, cx_thresh2):
        """
        Build the gene-pair-specific whisper network on top of the
        spatial neighbor graph and store it in self.graph_whisper.
        """
        graph_gene_pair = self.whisper_network(
            df_cx1,
            df_cx2,
            cx_thresh1,
            cx_thresh2,
        )
        # Elementwise AND between spatial neighbors and gene-expression mask
        self.graph_whisper = np.multiply(self.graph_dist, graph_gene_pair)

    def whisper_network(self, g1_exp, g2_exp, cx_thresh1, cx_thresh2):
        """
        Build a gene1→gene2 "whisper network" purely from expression thresholds.

        whisper_net[i, j] = True if cell i expresses g1 > cx_thresh1
        and cell j expresses g2 > cx_thresh2.
        """
        # Ensure column vectors
        g1_exp_bool = np.zeros(g1_exp.shape, dtype=bool).reshape(-1, 1)
        g2_exp_bool = np.zeros(g2_exp.shape, dtype=bool).reshape(-1, 1)

        g1_exp_bool[g1_exp > cx_thresh1] = True
        g2_exp_bool[g2_exp > cx_thresh2] = True

        whisper_net = np.matmul(g1_exp_bool, g2_exp_bool.T)
        return whisper_net



def create_or_load_whisper_graph(
    df_spatial,
    annot_df,
    annot,
    f_gj_object="results/graph_proximal_base_ht.object",
):
    """
    Create (or load) a BuildWhisperNetwork object.

    Parameters
    ----------
    df_spatial : array-like, shape (n_cells, n_dims)
    annot_df : pandas.DataFrame
        Annotation DataFrame with one column.
    annot : str
        Column name in annot_df to use as annotation labels.
    f_gj_object : str or None
        Path to a pickled BuildWhisperNetwork object.
        If None, a new object is always created.
        If the file exists, it is loaded instead of recomputed.

    Returns
    -------
    gj : BuildWhisperNetwork
    """
    if f_gj_object is None:
        gj = BuildWhisperNetwork(df_spatial, annot_df, annot, num_neighbor=5)
    elif os.path.exists(f_gj_object):
        print("loading neighbor object")
        gj = pkl_load(f_gj_object)
    else:
        gj = BuildWhisperNetwork(df_spatial, annot_df, annot, num_neighbor=5)
        print("saving neighbor object")
        pkl_save(gj, f_gj_object)
    return gj


# Optional: nicer name
def create_whisper_graph(
    df_spatial,
    annot_df,
    annot,
    f_object="results/graph_proximal_base_ht.object",
):
    """Alias of create_or_load_whisper_graph with a CellWHISPER-style name."""
    return create_or_load_whisper_graph(df_spatial, annot_df, annot, f_object)


# ----------------------------------------------------------------------
# Analytical null model (mean and std of edge counts)
# ----------------------------------------------------------------------
def null_prob(g1_exp, g2_exp, cx_thresh1, cx_thresh2, cell_type_list, annot_list):
    """
    Compute null probabilities p1, p2 per cell type and p1*p2 matrix.

    Parameters
    ----------
    g1_exp, g2_exp : array-like, shape (n_cells, 1) or (n_cells,)
        Expression of gene1 and gene2.
    cx_thresh1, cx_thresh2 : float
        Thresholds for binarizing gene1/gene2 expression.
    cell_type_list : list of str
        Unique ordered list of cell-type labels.
    annot_list : array-like of str, shape (n_cells,)
        Cell-type label for each cell, must be elements in cell_type_list.

    Returns
    -------
    p1_p2_mat : ndarray, shape (n_cell_types, n_cell_types)
        Outer product p1_list * p2_list^T.
    p1_list : ndarray, shape (n_cell_types, 1)
    p2_list : ndarray, shape (n_cell_types, 1)
    g1_exp_bool : ndarray of bool, shape (n_cells, 1)
    g2_exp_bool : ndarray of bool, shape (n_cells, 1)
    """
    # Boolean expression indicators
    g1_exp_bool = np.zeros(g1_exp.shape, dtype=bool).reshape(-1, 1)
    g2_exp_bool = np.zeros(g2_exp.shape, dtype=bool).reshape(-1, 1)

    g1_exp_bool[g1_exp > cx_thresh1] = True
    g2_exp_bool[g2_exp > cx_thresh2] = True

    # probabilities per cell type
    p1_list = np.ones(len(cell_type_list)).reshape(-1, 1)
    p2_list = np.ones(len(cell_type_list)).reshape(-1, 1)

    cell_type_index_list = np.array(
        [cell_type_list.index(i) for i in annot_list]
    )

    for i in range(len(cell_type_list)):
        cells_index_ct_list = cell_type_index_list == i
        temp_cx1 = g1_exp_bool[cells_index_ct_list]
        temp_cx2 = g2_exp_bool[cells_index_ct_list]

        # fraction of expressing cells within this cell type
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
    """
    Compute analytical standard deviation of edge counts between
    cell-type pairs under the non-i.i.d. null model.

    Parameters
    ----------
    adj_mat : ndarray of bool, shape (n_cells, n_cells)
        Spatial adjacency matrix (graph_dist).
    cell_type_list : list of str
    annot_list : array-like of str, shape (n_cells,)
    p1_list, p2_list : ndarray, shape (n_cell_types, 1)
        Per-cell-type probabilities for gene1/gene2.
    g1_exp_bool, g2_exp_bool : ndarray of bool, shape (n_cells, 1)
        Boolean expression flags.
    num_prox_ct_pair : ndarray, shape (n_cell_types, n_cell_types)
        Total number of proximal pairs per cell-type pair.
    mean_mat_curr : ndarray, shape (n_cell_types, n_cell_types)
        Expected number of edges under null: num_prox_ct_pair * p1*p2.

    Returns
    -------
    std_ct_pair : ndarray, shape (n_cell_types, n_cell_types)
        Standard deviation of edge counts per cell-type pair.
        A small MIN_VALUE is added for numerical stability.
    """
    cell_type_index_list = np.array(
        [cell_type_list.index(i) for i in annot_list]
    )

    n_ct = len(cell_type_list)
    std_ct_pair = -100 * np.ones((n_ct, n_ct))  # preserved initialization
    cells_index_ct_list = [-1] * n_ct

    # Flatten expression flags for easier indexing
    g1_exp_bool = g1_exp_bool.reshape(-1)
    g2_exp_bool = g2_exp_bool.reshape(-1)

    # Precompute boolean masks per cell type
    for i in range(n_ct):
        cells_index_ct_list[i] = cell_type_index_list == i

    for i in range(n_ct):
        temp = adj_mat[cells_index_ct_list[i], :]

        for j in range(n_ct):
            adj_mat_ct = temp[:, cells_index_ct_list[j]]

            # case 1: pairs sharing a gene-1-expressing cell
            deg_ct = adj_mat_ct.sum(axis=1)
            deg_ct[g1_exp_bool[cells_index_ct_list[i]] == 0] = 0
            deg_ct[deg_ct < 2] = 0
            num_case1 = (deg_ct * (deg_ct - 1) / 2).sum()

            # case 2: pairs sharing a gene-2-expressing cell
            deg_ct = adj_mat_ct.sum(axis=0)
            deg_ct[g2_exp_bool[cells_index_ct_list[j]] == 0] = 0
            deg_ct[deg_ct < 2] = 0
            num_case2 = (deg_ct * (deg_ct - 1) / 2).sum()

            if i != j:
                curr_var = (
                    num_case1 * p1_list[i] * (p2_list[j] ** 2)
                    + num_case2 * (p1_list[i] ** 2) * p2_list[j]
                    + (num_prox_ct_pair[i][j] ** 2 - num_case1 - num_case2)
                    * (p1_list[i] ** 2)
                    * (p2_list[j] ** 2)
                )
            else:
                curr_var = (
                    num_case1 * p1_list[i] * (p2_list[j] ** 2)
                    + (num_prox_ct_pair[i][j] ** 2 - num_case1)
                    * (p1_list[i] ** 2)
                    * (p2_list[j] ** 2)
                )

            std_ct_pair[i, j] = np.sqrt(
                curr_var + mean_mat_curr[i, j] - mean_mat_curr[i, j] ** 2
            )

    return std_ct_pair + MIN_VALUE


def cal_ct_pair_count(adj_mat, cell_type_list, annot_list):
    """
    Count edges between cell-type pairs in a given adjacency matrix.

    Parameters
    ----------
    adj_mat : ndarray of bool, shape (n_cells, n_cells)
        Adjacency matrix (either graph_dist or graph_whisper).
    cell_type_list : list of str
        Ordered list of unique cell types.
    annot_list : array-like of str, shape (n_cells,)
        Annotations for each cell.

    Returns
    -------
    ct_pair_count : ndarray, shape (n_cell_types, n_cell_types)
        For i != j: total number of edges between cells of type i and j.
        For i == j: number of edges among cells of type i (upper triangle).
    """
    cell_type_index_list = np.array(
        [cell_type_list.index(i) for i in annot_list]
    )
    n_ct = len(cell_type_list)

    # start with ones (as in original code)
    ct_pair_count = np.ones((n_ct, n_ct), dtype=int)

    cells_index_ct_list = [-1] * n_ct
    for i in range(n_ct):
        cells_index_ct_list[i] = cell_type_index_list == i

    for i in range(n_ct):
        temp = adj_mat[cells_index_ct_list[i], :]
        for j in range(n_ct):
            adj_mat_ct = temp[:, cells_index_ct_list[j]]
            if i != j:
                ct_pair_count[i, j] = adj_mat_ct.sum()
            else:
                # count each undirected edge once
                ct_pair_count[i, i] = np.triu(adj_mat_ct, 1).sum()  # unit-test comment

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
    f_gj_object="results/graph_proximal_base_ss.object",
):
    """
    Run CellWHISPER statistical test for a single signaling-gene pair
    (gap junction or ligand–receptor), returning counts, mean, std, z-scores.

    Parameters
    ----------
    df_spatial : array-like, shape (n_cells, n_dims)
        Spatial coordinates.
    df_cx1, df_cx2 : array-like, shape (n_cells, 1) or (n_cells,)
        Expression vectors for gene1 and gene2.
    annot_df : pandas.DataFrame
        DataFrame with a single column containing cell-type annotations.
    percentile : int, default 75
        Percentile used as global threshold for binarizing gene expression.
    mode : {"iid", "non_iid"}, default "non_iid"
        If "iid", use simple Binomial variance p(1-p).
        If "non_iid", use analytical variance from null_std that
        accounts for shared neighbors.
    annot : str, default "annotation"
        Column name in annot_df with cell-type labels.
    f_gj_object : str or None
        Path to neighbor-graph pickle file or None to always recompute.

    Returns
    -------
    num_gj_ct_pair : ndarray, shape (n_ct, n_ct)
        Observed whisper edges per cell-type pair.
    num_prox_ct_pair : ndarray, shape (n_ct, n_ct)
        Total proximal cell pairs per cell-type pair.
    z_score_gj_pair : ndarray, shape (n_ct, n_ct)
        Z-scores of observed vs. null.
    mean_mat_curr : ndarray, shape (n_ct, n_ct)
        Null mean of whisper edges.
    std_mat_curr : ndarray, shape (n_ct, n_ct)
        Null standard deviation of whisper edges.
    """
    # Load or construct spatial neighbor graph
    gj = create_whisper_graph(df_spatial, annot_df, annot, f_gj_object)

    # Global thresholds based on percentiles across all cells
    cx_thresh1 = np.percentile(df_cx1, percentile)
    cx_thresh2 = np.percentile(df_cx2, percentile)
    print(f"thresh 1 {cx_thresh1} thresh 2 {cx_thresh2}")

    # Build gene-filtered whisper network on top of spatial graph
    gj.create_whisper_network(df_cx1, df_cx2, cx_thresh1, cx_thresh2)

    # Cell-type list and annotation vector
    cell_type_list = list(np.unique(annot_df.values))

    (
        p1_p2_mat,
        p1_list,
        p2_list,
        g1_exp_bool,
        g2_exp_bool,
    ) = null_prob(
        df_cx1,
        df_cx2,
        cx_thresh1,
        cx_thresh2,
        cell_type_list,
        annot_df.values,
    )

    # Observed whisper edges and total proximal pairs per cell-type pair
    num_gj_ct_pair = cal_ct_pair_count(
        gj.graph_whisper,
        cell_type_list,
        annot_df.values,
    )
    num_prox_ct_pair = cal_ct_pair_count(
        gj.graph_dist,
        cell_type_list,
        annot_df.values,
    )

    # Null mean
    mean_mat_curr = np.multiply(num_prox_ct_pair, p1_p2_mat)

    # Null std
    if mode == "iid":
        std_mat_curr = np.sqrt(
            np.multiply(mean_mat_curr, (1 - p1_p2_mat)) + MIN_VALUE
        )
    else:
        std_mat_curr = null_std(
            gj.,
            cell_type_list,
            annot_df.values,
            p1_list,
            p2_list,
            g1_exp_bool,
            g2_exp_bool,
            num_prox_ct_pair,
            mean_mat_curr,
        )

    # Z-score (with tiny epsilon to avoid divide-by-zero)
    z_score_gj_pair = (num_gj_ct_pair - mean_mat_curr) / (
        std_mat_curr + 1e-32
    )

    return (
        num_gj_ct_pair,
        num_prox_ct_pair,
        z_score_gj_pair,
        mean_mat_curr,
        std_mat_curr,
    )

def run_cellwhisper_pair_adata(
    adata,
    gene1,
    gene2,
    annot="annotation",
    spatial_key="spatial",
    percentile=75,
    mode="non_iid",
    graph_cache="results/graph_proximal_base_ss.object",
    store_key=None,
):
    """
    Run CellWHISPER on a single gene pair using an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Must have adata.obsm[spatial_key] and adata.obs[annot].
    gene1, gene2 : str
        Gene names present in adata.var_names.
    annot : str, default "annotation"
        Cell-type column in adata.obs.
    spatial_key : str, default "spatial"
        Key in adata.obsm containing spatial coordinates.
    percentile : int, default 75
        Global expression threshold percentile.
    mode : {"iid", "non_iid"}, default "non_iid"
        "non_iid" for gap junction / strong null,
        "iid" for simple binomial null (e.g. LR).
    graph_cache : str or None
        Path to neighbor-graph pickle. If None, graph is recomputed
        every time. If a path is given, it will be created/loaded via
        create_whisper_graph.
    store_key : str or None
        If not None, results will be stored in
        adata.uns[store_key]. If None, nothing is stored.

    Returns
    -------
    result : dict
        {
          "gene1": gene1,
          "gene2": gene2,
          "cell_types": [...],
          "num_whisper": (n_ct, n_ct) array,
          "num_prox": (n_ct, n_ct) array,
          "z_score": (n_ct, n_ct) array,
          "mean": (n_ct, n_ct) array,
          "std": (n_ct, n_ct) array,
          "percentile": percentile,
          "mode": mode,
        }
    """
    # Extract basic pieces
    df_spatial = adata.obsm[spatial_key]
    annot_df = adata.obs[annot].to_frame()

    # Expression vectors (keep .toarray() for full compatibility)
    df_cx1 = adata[:, gene1].X.toarray()
    df_cx2 = adata[:, gene2].X.toarray()

    (
        num_whisper_ct_pair,
        num_prox_ct_pair,
        z_score_ct_pair,
        mean_mat_curr,
        std_mat_curr,
    ) = run_cellwhisper_pair(
        df_spatial=df_spatial,
        df_cx1=df_cx1,
        df_cx2=df_cx2,
        annot_df=annot_df,
        percentile=percentile,
        mode=mode,
        annot=annot,
        f_gj_object=graph_cache,
    )

    cell_type_list = list(np.unique(annot_df.values))

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
    }

    if store_key is not None:
        adata.uns[store_key] = result

    return result


