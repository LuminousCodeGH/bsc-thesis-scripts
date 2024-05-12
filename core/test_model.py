import anndata as ad
import pandas as pd
from collections.abc import Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin
from core.kfold_cv import kfold_cv
from core.normalize import *
from scipy.stats import ranksums
from statsmodels.stats.multitest import fdrcorrection


def test_model(adata: ad.AnnData, 
               model: ClassifierMixin, 
               y: pd.Series, 
               norm_layer: str='norm', 
               significance_col: str='significant',
               sig_filter: bool=False) -> pd.DataFrame:
    if sig_filter:
        X = pd.DataFrame(adata.layers[norm_layer][:, adata.var[significance_col] == True])
    else:
        X = pd.DataFrame(adata.layers[norm_layer])
    
    cv_res = kfold_cv(model, X, y)

    return pd.DataFrame(cv_res)


def test_abundance(adata: ad.AnnData,
                   test_func: Callable=ranksums,
                   corr_func: Callable=fdrcorrection,
                   norm_layer: str='norm',
                   alpha: float=0.05,
                   X_filter: pd.Series=None,
                   Y_filter: pd.Series=None) -> None:
    if X_filter is None:
        X = adata.layers[norm_layer][adata.obs['cogdx'] == 1]
    else:
        X = adata.layers[norm_layer][X_filter]

    if Y_filter is None:
        Y = adata.layers[norm_layer][(adata.obs['cogdx'] == 4) | (adata.obs['cogdx'] == 5)]
    else:
        Y = adata.layers[norm_layer][Y_filter]
    res = test_func(X, Y)

    fdr_correction_res = corr_func(res.pvalue, alpha)
    adata.var['significant'] = fdr_correction_res[0]
    adata.var['corr_pvalue'] = fdr_correction_res[1]

    adata.var.sort_values('corr_pvalue')


def test_model_from_scratch(adata: ad.AnnData, 
                             norm_func: Callable=normalize_minmax, 
                             layer_name: str='norm',
                             model: ClassifierMixin=RandomForestClassifier(class_weight='balanced'), 
                             alpha: float=0.05,
                             y: pd.Series=None,
                             sig_filter: bool=False) -> pd.DataFrame:
    '''Default test is between cogdx=1 and cogdx=4/5 on a RandomForestClassifier'''
    norm_func(adata, layer_name=layer_name)
    test_abundance(adata, ranksums, fdrcorrection, norm_layer=layer_name, alpha=alpha)

    return test_model(adata, model, y, norm_layer=layer_name, sig_filter=sig_filter)


def optimize_normalization(X: ad.AnnData,
                           y: pd.Series,
                           classifier,
                           normalizations: list[Callable[[ad.AnnData], None]] = [normalize_l1, 
                                                                                 normalize_l2, 
                                                                                 normalize_minmax, 
                                                                                 normalize_robust, 
                                                                                 normalize_tmm, 
                                                                                 normalize_mrn], 
                           layer_names: list[str] = ['l1', 'l2', 'minmax', 'robust', 'tmm', 'mrn']) -> pd.DataFrame:
    results: pd.DataFrame
    results = pd.concat([test_model_from_scratch(X, norm_func, layer, classifier, y=y.cat.codes).assign(_h=layer) 
                             for norm_func, layer in zip(normalizations, layer_names)])
    return results
