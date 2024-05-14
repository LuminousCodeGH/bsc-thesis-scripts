import scanpy as sc
import anndata as ad
import numpy as np
from sklearn.preprocessing import normalize, minmax_scale, robust_scale
from conorm import tmm, mrn
from collections.abc import Callable


def normalize_l1(adata: ad.AnnData, layer_name: str='norm', transformation_func: Callable[[np.ndarray], None]=sc.pp.log1p, inplace=True, reference: str=None) -> None | ad.AnnData:
    '''https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html#normalization'''
    result: ad.AnnData | None = None
    if not inplace:
        adata = adata.copy()
        result = adata
    adata.layers[layer_name] = adata.X.copy()
    sc.pp.normalize_total(adata, layer=layer_name, target_sum=10000)  # This is a simple L1 normalization
    if transformation_func is not None:
        transformation_func(adata.layers[layer_name])
    adata.layers[layer_name][np.isnan(adata.layers[layer_name])] = 0.0
    return result


def normalize_l2(adata: ad.AnnData, layer_name: str='norm', transformation_func: Callable[[np.ndarray], None]=sc.pp.log1p, inplace=True, reference: str=None) -> None | ad.AnnData:
    '''https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html#normalization'''
    result: ad.AnnData | None = None
    if not inplace:
        adata = adata.copy()
        result = adata
    adata.layers[layer_name] = adata.X.copy()
    adata.layers[layer_name] = normalize(adata.layers[layer_name], 'l2', axis=1) * 10000
    if transformation_func is not None:
        transformation_func(adata.layers[layer_name])
    adata.layers[layer_name][np.isnan(adata.layers[layer_name])] = 0.0
    return result


def normalize_minmax(adata: ad.AnnData, layer_name: str='norm', transformation_func: Callable[[np.ndarray], None]=sc.pp.log1p, inplace=True, reference: str=None) -> None | ad.AnnData:
    '''https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html#sklearn.preprocessing.minmax_scale'''
    result: ad.AnnData | None = None
    if not inplace:
        adata = adata.copy()
        result = adata
    adata.layers[layer_name] = adata.X.copy()
    adata.layers[layer_name] = minmax_scale(adata.layers[layer_name], axis=1) * 10000
    if transformation_func is not None:
        transformation_func(adata.layers[layer_name])
    adata.layers[layer_name][np.isnan(adata.layers[layer_name])] = 0.0
    return result


def normalize_robust(adata: ad.AnnData, layer_name: str='norm', transformation_func: Callable[[np.ndarray], None]=sc.pp.log1p, inplace=True, reference: str=None) -> None | ad.AnnData:
    '''https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html'''
    result: ad.AnnData | None = None
    if not inplace:
        adata = adata.copy()
        result = adata
    adata.layers[layer_name] = adata.X.copy()
    adata.layers[layer_name] = robust_scale(adata.layers[layer_name], axis=1, with_centering=False) * 10000
    if transformation_func is not None:
        transformation_func(adata.layers[layer_name])
    adata.layers[layer_name][np.isnan(adata.layers[layer_name])] = 0.0
    return result


def normalize_tmm(adata: ad.AnnData, layer_name: str='norm', transformation_func: Callable[[np.ndarray], None]=sc.pp.log1p, inplace=True, reference: str=None) -> None | ad.AnnData:
    '''https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7710733/ // https://pypi.org/project/conorm/'''
    result: ad.AnnData | None = None
    if not inplace:
        adata = adata.copy()
        result = adata
    adata.layers[layer_name] = adata.X.copy()

    adata.layers[layer_name] = tmm(adata.layers[layer_name].T).T  # Scaling seems to happen automatically
    if transformation_func is not None:
        transformation_func(adata.layers[layer_name])
    adata.layers[layer_name][np.isnan(adata.layers[layer_name])] = 0.0
    return result

def normalize_mrn(adata: ad.AnnData, 
                  layer_name: str='norm', 
                  transformation_func: Callable[[np.ndarray], None]=sc.pp.log1p, 
                  inplace=True,
                  reference: str=None) -> None | ad.AnnData:
    '''https://pypi.org/project/conorm/'''
    result: ad.AnnData | None = None
    if not inplace:
        adata = adata.copy()
        result = adata
    adata.layers[layer_name] = adata.X.copy()

    adata.layers[layer_name] = mrn(adata.layers[layer_name].T).T  # Scaling seems to happen automatically
    if transformation_func is not None:
        transformation_func(adata.layers[layer_name])
    adata.layers[layer_name][np.isnan(adata.layers[layer_name])] = 0.0
    return result

def normalize_by_reference(adata: ad.AnnData,  
                           layer_name: str='norm', 
                           transformation_func: Callable[[np.ndarray], None]=sc.pp.log1p, 
                           inplace=True,
                           reference: str='OPCs') -> None | ad.AnnData:
    result: ad.AnnData | None = None
    if reference is None:
        raise ValueError('Reference cell type is unset!')

    if not inplace:
        adata = adata.copy()
        result = adata
    adata.layers[layer_name] = adata.X.copy()
    for row in adata.layers[layer_name]:
        ref_idx = adata.var_names.get_loc(reference)
        row = row / row[ref_idx]
    if transformation_func is not None:
            transformation_func(adata.layers[layer_name])
    adata.layers[layer_name][np.isnan(adata.layers[layer_name])] = 0.0
    return result
