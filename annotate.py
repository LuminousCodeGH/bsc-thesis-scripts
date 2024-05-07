import anndata as ad


def _categorize_by(adata: ad.AnnData, by: str, map: dict[int, str], col_name: str, inplace=True) -> None | ad.AnnData:
    assert all(u_entry in map.keys() for u_entry in adata.obs[by].unique()), \
        f'The map is missing entries! \nmap: {map.keys()}\nentries: {adata.obs[by].unique().tolist()}'
    
    result: ad.AnnData | None = None
    if not inplace:
        adata = adata.copy()
        result = adata

    adata.obs[col_name] = adata.obs.apply(lambda row: map[row[by]], axis=1)
    return result


def categorize_cogdx(adata: ad.AnnData, map: dict[int, str], col_name: str='cogdx_class', inplace=True) -> None | ad.AnnData:
    return _categorize_by(adata, 'cogdx', map, col_name, inplace)