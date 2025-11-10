import scanpy as sc
import pandas as pd
import numpy as np
import anndata
import pickle
from scipy import stats
from matplotlib import pyplot as plt



def pkl_load(f):
    with open(f,'rb') as f:
        data = pickle.load(f)
    return data

def pkl_save(data,f):
    print(f'saving in {f}')
    with open(f, 'wb') as f:
        pickle.dump(data, f)