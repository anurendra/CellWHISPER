import scanpy as sc
import pandas as pd
import numpy as np
import anndata
import pickle
from scipy import stats
from matplotlib import pyplot as plt
import scipy.sparse as sp



def pkl_load(f):
    with open(f,'rb') as f:
        data = pickle.load(f)
    return data

def pkl_save(data,f):
    print(f'saving in {f}')
    with open(f, 'wb') as f:
        pickle.dump(data, f)

def unstack_cellwhisper(res, symmetric=None, min_edges=0):
    """
    Convert CellWHISPER result dict (CTxCT matrices) into a tidy quadruplet dataframe.
    """
    g1, g2 = res['gene1'], res['gene2']
    cts = res['cell_types']
    n = len(cts)
    if symmetric is None:
        symmetric = (g1 == g2)
    
    if symmetric:
        idx = np.triu_indices(n)
    else:
        # all ordered pairs: ct1 expresses gene1 (ligand), ct2 expresses gene2 (receptor)
        idx = np.unravel_index(np.arange(n * n), (n, n))
    
    df = pd.DataFrame({
        'ct1': [cts[i] for i in idx[0]],
        'ct2': [cts[j] for j in idx[1]],
        'gene1': g1,
        'gene2': g2,
        'n_whisper_edges': np.asarray(res['num_whisper'])[idx],
        'n_prox_edges': np.asarray(res['num_prox'])[idx],
        'null_mean': np.asarray(res['mean'])[idx],
        'null_std': np.asarray(res['std'])[idx],
        'z_score': np.asarray(res['z_score'])[idx],
    })
    df['quadruplet'] = df['ct1'] + ' | ' + df['ct2'] + ' | ' + g1 + '-' + g2
    df = df.set_index('quadruplet')
    if min_edges > 0:
        df = df[df['n_whisper_edges'] >= min_edges]
    return df

def edge_segs(G, annot, ct1, ct2, pos):
    """
    Whisper edges of one quadruplet as line segments for plotting.
    G: whisper network (sparse or dense adjacency), annot: 1D array of cell-type labels,
    pos: (n_cells, 2) coordinates. Returns array of shape (n_edges, 2, 2).
    Counting matches cal_ct_pair_count: for ct1 == ct2 each edge is taken once.
    """
    Cm = sp.coo_matrix(G)
    annot = np.asarray(annot).astype(str)
    m = (annot[Cm.row] == ct1) & (annot[Cm.col] == ct2)
    if ct1 == ct2:
        m &= Cm.row < Cm.col
    return np.stack([pos[Cm.row[m]], pos[Cm.col[m]]], axis=1)


def plot_whisper_network(ax, segs, pos, title=None):
    """
    Cells in grey, whisper edges in blue, density of edge midpoints in red.
    """
    import seaborn as sns
    from matplotlib.collections import LineCollection
    mid = segs.mean(axis=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=1.5, c='#e6e6e6', rasterized=True)
    if len(segs) > 1:
        sns.kdeplot(x=mid[:, 0], y=mid[:, 1], cmap='Reds', fill=True, bw_adjust=0.3,
                    alpha=0.6, levels=100, thresh=0.02, ax=ax)
    ax.add_collection(LineCollection(segs, colors='b', linewidths=0.8, alpha=0.8))
    ax.set_aspect('equal'); ax.set_axis_off()
    if title:
        ax.set_title(title)

