import anndata as ad
import pandas as pd
import numpy as np

from collections.abc import Collection
from typing import Literal


class Categorizer:
    COGDX_MAP = {1: 'H', 2: 'M', 3: 'M', 4: 'AD', 5: 'AD', 6: 'O'}
    CERADSC_MAP = {1: 'Y', 2: 'Y', 3: 'N', 4: 'N'}

    def __init__(self, add_catcodes: bool) -> None:
        self.add_catcodes = add_catcodes

    def categorize(self, 
                   adata: ad.AnnData, 
                   column: str, 
                   map: dict[str | int, str], 
                   inplace=True) -> None | ad.AnnData:
        if not inplace:
            adata = adata.copy()
        assert all(u_entry in map.keys() for u_entry in adata.obs[column].unique()), \
            f'The map is missing entries! \nmap: {map.keys()}\nentries: {adata.obs[column].unique().tolist()}'
        
        adata.obs[f'{column}_cat'] = adata.obs.apply(lambda row: map[row[column]], axis=1).astype('category')
        if self.add_catcodes:
            adata.obs[f'{column}_catcode'] = adata.obs[f'{column}_cat'].cat.codes
        if not inplace:
            return adata
        
    def combine_ct(self, 
                   adata: ad.AnnData, 
                   cell_types: Collection[str], 
                   combine_leftovers: Literal['all'] | Collection[str] | None, 
                   leftover_name: str='Other') -> ad.AnnData:
        result = adata.copy()
        columns = []
        X_new = []

        _comb_left: int = 1 if combine_leftovers != None else 0
        for i in range(len(cell_types) + _comb_left):
            if i == len(cell_types) and combine_leftovers is not None:
                # Handle the 'other' cell types
                columns.append(leftover_name)
                if combine_leftovers == 'all':
                    X_new.append(result.X[:, ~result.var.index.isin(cell_types)].tolist())
                else:
                    X_new(result.X[:, result.var.index.isin(combine_leftovers)].tolist())
                break
            columns.append(cell_types[i])
            X_new.append(result.X[:, result.var.index.str.startswith(columns[i])].tolist())

        # Sum all cell types together for each major group
        for j in range(len(X_new)):
            for i in range(len(X_new[j])):
                X_new[j][i] = sum(count for count in list(X_new[j][i]))
        X_new = np.array(X_new)
        print(f'{X_new.shape=}')
        result = ad.AnnData(X_new.T, result.obs, pd.DataFrame(index=columns))
        return result
    