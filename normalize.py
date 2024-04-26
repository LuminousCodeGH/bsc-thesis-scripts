import scanpy as sc
import anndata as ad
from sklearn.preprocessing import normalize, minmax_scale, quantile_transform
from conorm import tmm, quartile


def normalize_log1p(adata: ad.AnnData, layer_name: str='norm', inplace=True) -> None | ad.AnnData:
    '''https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html#normalization'''
    result: ad.AnnData | None = None
    if not inplace:
        adata = adata.copy()
        result = adata
    adata.layers[layer_name] = adata.X.copy()
    sc.pp.normalize_total(adata, layer=layer_name)
    sc.pp.log1p(adata, layer=layer_name)
    return result


def normalize_unit(adata: ad.AnnData, layer_name: str='norm', norm_type: str='l2', inplace=True) -> None | ad.AnnData:
    '''https://genomebiology.biomedcentral.com/articles/10.1186/gb-2010-11-3-r25'''
    result: ad.AnnData | None = None
    if not inplace:
        adata = adata.copy()
        result = adata
    adata.layers[layer_name] = adata.X.copy()
    adata.layers[layer_name] = normalize(adata.layers[layer_name], norm_type)
    assert adata.layers[layer_name].shape == adata.X.shape
    return result


def normalize_unit_tmm(adata: ad.AnnData, layer_name: str='norm', inplace=True) -> None | ad.AnnData:
    '''https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7710733/'''
    result: ad.AnnData | None = None
    if not inplace:
        adata = adata.copy()
        result = adata
    adata.layers[layer_name] = adata.X.copy()

    # For each sample, scale to relative abundance
    adata.layers[layer_name] = normalize(adata.layers[layer_name])

    # Run TMM
    adata.layers[layer_name] = tmm(adata.layers[layer_name])
    sc.pp.log1p(adata, layer=layer_name)
    assert adata.layers[layer_name].shape == adata.X.shape
    return result


def normalize_quartile_tmm(adata: ad.AnnData, layer_name: str='norm', inplace=True) -> None | ad.AnnData:
    result: ad.AnnData | None = None
    if not inplace:
        adata = adata.copy()
        result = adata
    adata.layers[layer_name] = adata.X.copy()

    adata.layers[layer_name] = quartile(adata.layers[layer_name], 'upper', 'TMM')
    assert adata.layers[layer_name].shape == adata.X.shape
    return result


def normalize_minmax(adata: ad.AnnData, layer_name: str='norm', inplace=True) -> None | ad.AnnData:
    '''https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html#sklearn.preprocessing.minmax_scale'''
    result: ad.AnnData | None = None
    if not inplace:
        adata = adata.copy()
        result = adata
    adata.layers[layer_name] = adata.X.copy()
    adata.layers[layer_name] = minmax_scale(adata.layers[layer_name])
    assert adata.layers[layer_name].shape == adata.X.shape
    return result


def normalize_quantile(adata: ad.AnnData, layer_name: str='norm', inplace=True) -> None | ad.AnnData:
    '''https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html'''
    result: ad.AnnData | None = None
    if not inplace:
        adata = adata.copy()
        result = adata
    adata.layers[layer_name] = adata.X.copy()
    adata.layers[layer_name] = quantile_transform(adata.layers[layer_name], n_quantiles=int(len(adata.X)**0.5))
    assert adata.layers[layer_name].shape == adata.X.shape
    return result