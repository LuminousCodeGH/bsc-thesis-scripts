import anndata as ad
import pandas as pd
from collections.abc import Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin
from kfold_cv import kfold_cv
from normalize import normalize_minmax
from scipy.stats import ranksums
from statsmodels.stats.multitest import fdrcorrection


def test_model(adata: ad.AnnData, 
                model: ClassifierMixin=RandomForestClassifier(class_weight='balanced'), 
                y: pd.Series=None, 
                norm_layer:str='norm', 
                significance_col: str='significant',
                sig_filter: bool=False) -> pd.DataFrame:
    if sig_filter:
        X = pd.DataFrame(adata.layers[norm_layer][:, adata.var[significance_col] == True])
    else:
        X = pd.DataFrame(adata.layers[norm_layer])

    if y is None:
        y = (adata.obs['cogdx'] == 5) | (adata.obs['cogdx'] == 4)

    cv_res = kfold_cv(model, X, y)
    adata.uns['model_training_results'] = cv_res

    return pd.DataFrame(cv_res)


def test_model_from_scratch(adata: ad.AnnData, 
                             norm_func: Callable=normalize_minmax, 
                             model: ClassifierMixin=RandomForestClassifier(class_weight='balanced'), 
                             alpha: float=0.05,
                             Y_filter: pd.Series=None,
                             sig_filter: bool=False) -> pd.DataFrame:
    '''Default test is between cogdx=1 and cogdx=4/5 on a RandomForestClassifier'''
    norm_func(adata)
    X = adata.layers['norm'][adata.obs['cogdx'] == 1]
    if Y_filter is None:
        Y = adata.layers['norm'][(adata.obs['cogdx'] == 4) | (adata.obs['cogdx'] == 5)]
    else:
        Y = adata.layers['norm'][Y_filter]
    res = ranksums(X, Y)

    fdr_correction_res = fdrcorrection(res.pvalue, alpha, 'indep')
    adata.var['significant'] = fdr_correction_res[0]
    adata.var['corr_pvalue'] = fdr_correction_res[1]

    adata.var.sort_values('corr_pvalue')

    return test_model(adata, model, Y_filter)
