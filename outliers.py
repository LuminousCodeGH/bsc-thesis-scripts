import anndata as ad
import numpy as np
import pandas as pd
from collections.abc import Callable
from normalize import normalize_l1


def _calculate_ct_per(adata: ad.AnnData, 
                      cell_type: str, 
                      normalize_first: bool=True,
                      layer: str='X',
                      column_name: str=None, 
                      replace_nans:float=0,
                      replace_infs:float=999999,
                      averge_func: Callable[[np.ndarray], float]=np.median,
                      inplace: bool=True) -> pd.DataFrame | None:
    result = None
    if not inplace:
        adata = adata.copy()
        
    if normalize_first:
        X: np.ndarray = normalize_l1(adata, 'norm_fraction', None, inplace=False).layers['norm_fraction']
    elif not normalize_first and layer == 'X':
        X: np.ndarray = adata.X
    elif not normalize_first and layer != 'X':
        X: np.ndarray = adata.layers[layer]
    
    if column_name is None:
        column_name = f'{cell_type}_p_ctmedian'

    ct_filter = adata.var.index == cell_type
    res: list[float] = []
    for row in X:
        assert float('nan') not in row
        avg_metric = averge_func(row[~ct_filter])
        ct_count = row[ct_filter][0]
        cpm = ct_count / avg_metric

        if ct_count == 0 and avg_metric == 0:
            print('Encountered NaN: ')
            print(f'avg_metric={avg_metric:.3f}, ct_count={ct_count:.3f}', end=' => ')
            print(f'count/avg_metric={cpm:.3f}')
            cpm = replace_nans
            print(f'Replacing with: {replace_nans}')

        elif cpm == float('inf'):
            print('Encountered infinity: ')
            print(f'avg_metric={avg_metric:.3f}, ct_count={ct_count:.3f}', end=' => ')
            print(f'count/avg_metric={cpm:.3f}')
            cpm = replace_infs
            print(f'Replacing with: {replace_infs}')

        res.append(cpm)

    if not inplace:
        result = pd.DataFrame(res, index=adata.obs_names, columns=[column_name])
    else:
        adata.obs[column_name] = pd.Series(res, index=adata.obs_names)

    return result


def calculate_ct_per_median(adata: ad.AnnData, 
                            cell_type: str, 
                            normalize_first: bool=True,
                            layer: str='X', 
                            column_name: str=None, 
                            replace_nans:float=0,
                            replace_infs:float=999999, 
                            inplace: bool=True) -> pd.DataFrame | None:
    _calculate_ct_per(adata, cell_type, normalize_first, layer, column_name, replace_nans, replace_infs, np.median, inplace)


def calculate_ct_per_mean(adata: ad.AnnData, 
                            cell_type: str, 
                            normalize_first: bool=True, 
                            layer: str='X', 
                            column_name: str=None, 
                            replace_nans:float=0,
                            replace_infs:float=999999, 
                            inplace: bool=True) -> pd.DataFrame | None:
    _calculate_ct_per(adata, cell_type, normalize_first, layer, column_name, replace_nans, replace_infs, np.mean, inplace)


def remove_outliers(adata: ad.AnnData, 
                    obs_metric: str,
                    threshold: float,
                    gt: bool=True,
                    inplace: bool=True) -> pd.DataFrame | None:
    result: ad.AnnData | None = None
    if not inplace:
        adata = adata.copy()
        result = adata
    
    if gt:
        adata = adata[~adata.obs[obs_metric] > threshold]
    else:
        adata = adata[~adata.obs[obs_metric] < threshold]

    return result