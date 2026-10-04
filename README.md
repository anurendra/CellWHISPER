# CellWHISPER

CellWHISPER infers contact-mediated cell-cell communication (gap junctions, juxtacrine ligand-receptor signaling) from single-cell resolution spatial transcriptomics.

For a quadruplet (cell type A, cell type B, gene 1, gene 2) it counts neighboring A-B cell pairs in which A expresses gene 1 and B expresses gene 2, the "whisper network", and scores that count against a null that shuffles cells within each cell type. The null keeps the spatial arrangement of every cell type, so a high z-score means co-expression at contacts beyond what the arrangement already predicts. Mean and variance of the null are computed in closed form, so no permutations are needed and a full ligand-receptor database can be scanned on tissues with tens of thousands of cells.

Manuscript: *CellWHISPER disentangles direct cell-cell communication from structural proximity* ([preprint](https://www.biorxiv.org/content/10.64898/2026.01.07.697982v2)).

## Install

Clone the repository and make sure these are available: `numpy`, `scipy`, `pandas`, `scikit-learn`, `scanpy`, `matplotlib`, `seaborn`.

```bash
git clone https://github.com/anurendra/CellWHISPER.git
cd CellWHISPER
```

The module is imported from the repository folder (`import cellwhisper as cw`); a pip package will follow.

## Quick start

Input is an AnnData object with normalized expression in `.X`, cell coordinates in `.obsm['spatial']` and cell-type labels in `.obs`.

```python
import cellwhisper as cw
from utils import unstack_cellwhisper

res = cw.run_cellwhisper_pair_adata(adata, gene1='GJA1', gene2='GJA1',
                                    annot='cell_type', spatial_key='spatial')
df = unstack_cellwhisper(res)                 # one row per cell-type pair
df[(df.z_score > 3) & (df.n_whisper_edges > 30)].sort_values('z_score', ascending=False)
```

For a ligand-receptor pair the test is directional (gene 1 in the first cell type, gene 2 in the second); pass `symmetric=False` to `unstack_cellwhisper` to keep all ordered pairs.

Main options of `run_cellwhisper_pair_adata`:

- `percentile` (75): per-gene binarization threshold; for sparse genes this is 0 and "expressed" means detected
- `num_neighbor` (5): k of the symmetrized k-NN contact graph; `max_distance` adds an optional distance cap
- `mode` (`non_iid`): variance accounting for contacts that share a cell; `iid` treats contacts as independent (same ranking, slightly higher z)
- `graph_cache`: path to store the neighbor graph when running many gene pairs on the same cells
- `return_graph`: also return the whisper network for plotting

## Tutorial

[`tutorials/01_skin_xenium.ipynb`](tutorials/01_skin_xenium.ipynb) runs the method end to end on a public 10x Xenium Prime 5K human skin dataset: loading, cell-type annotation including epidermal layers, a gap-junction pair, a ligand-receptor pair, and plots of where the contacts are.

Coming next: loaders for CellChatDB, CellPhoneDB and OmniPath contact pairs, and the latent variable model that summarizes a database scan into cell-type and gene preference maps.

## Note on cell-type annotation

The null conditions on the labels you provide. In tissues with spatial gradients (crypt axes, epidermal layers, cortical layers), positional states should be part of the annotation; otherwise a quadruplet confined to one zone is tested against the whole cell type.

## Citation

Kumar, Anurendra, Felix Rivera, Bhavay Aggarwal, Nicholas Zhang, Ahmet Coskun, and Saurabh Sinha. "CellWHISPER disentangles direct cell–cell communication from structural proximity." bioRxiv (2026): 2026-01. https://www.biorxiv.org/content/10.64898/2026.01.07.697982v2

## License

LICENSE_NAME