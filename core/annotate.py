import anndata as ad


COGDX_MAP = {1: 'H', 2: 'M', 3: 'M', 4: 'AD', 5: 'AD', 6: 'O'}


def _categorize_by(adata: ad.AnnData, by: str, map: dict[int, str], col_name: str, inplace=True) -> None | ad.AnnData:
    assert all(u_entry in map.keys() for u_entry in adata.obs[by].unique()), \
        f'The map is missing entries! \nmap: {map.keys()}\nentries: {adata.obs[by].unique().tolist()}'
    
    result: ad.AnnData | None = None
    if not inplace:
        adata = adata.copy()
        result = adata

    adata.obs[col_name] = adata.obs.apply(lambda row: map[row[by]], axis=1).astype('category')
    return result


def categorize_cogdx(adata: ad.AnnData, map: dict[int, str], col_name: str='cogdx_class', inplace=True) -> None | ad.AnnData:
    return _categorize_by(adata, 'cogdx', map, col_name, inplace)