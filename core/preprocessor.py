import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Callable, Collection
from typing import Literal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    COGDX_MAP = {1: 'H', 2: 'M', 3: 'M', 4: 'AD', 5: 'AD', 6: 'O'}

    def __init__(self) -> None:
        self._leiden_clusters: pd.Series = None
        self.n_components: int = None

    @property
    def leiden_clusters(self) -> pd.Series:
        return self._leiden_clusters

    @leiden_clusters.setter
    def leiden_clusters(self, value) -> None:
        raise AttributeError('Attribute is set automatically when running \'Preprocessor.analyze_umap\'!')
    
    def set_n_components(self, value: int) -> None:
        self.n_components = value

    def analyze_umap(self, adata: ad.AnnData, return_adata: bool=False) -> None:
        if self.n_components is None:
            raise AttributeError('The number of PCs to use is unset!')
        adata = self.show_umap(adata, None, return_adata=True)

        self._leiden_clusters = adata.obs['leiden']

        if return_adata:
            return adata
        
    def show_umap(self, adata: ad.AnnData, n_components: int=None, title_addition: str='', return_adata: bool=False) -> None | ad.AnnData:
        if n_components is None:
            n_components = self.n_components

        adata = adata.copy()
        adata.layers['umap_norm'] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=10000, layer='umap_norm')
        sc.pp.log1p(adata.layers['umap_norm'])
        sc.pp.pca(adata, n_components, layer='umap_norm')
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, flavor="igraph", n_iterations=4)
        ax = sc.pl.umap(adata, color="leiden", show=False)
        ax.set_title(f'UMAP of Leiden Clustered PCs (n={n_components}){title_addition}')
        ax.figure.show()

        if return_adata:
            return adata
        
    def show_pca(self, adata: ad.AnnData) -> None:
        adata = adata.copy()
        _n_comps: int = adata.var_names.size
        sc.pp.normalize_total(adata, target_sum=10000)
        X = StandardScaler().fit_transform(adata.X)
        model = PCA(n_components=_n_comps).fit(X)
        self._pca_screeplot(model)
        
    def remove_cluster(self, adata: ad.AnnData, cluster: str) -> ad.AnnData:
        if self.n_components is None:
            raise AttributeError('The number of PCs to use is unset!')
        if self.leiden_clusters is None:
            raise AttributeError('No leiden clustering has been performed yet. Run Preprocessor.analyze_umap first!')

        adata = adata[self.leiden_clusters != cluster, :]
        self.show_umap(adata, self.n_components, ', Post-removal')
        return adata
    
    def categorize(self, adata: ad.AnnData, column: str, map: dict[str | int, str], add_catcodes: bool=False, inplace=True) -> None | ad.AnnData:
        if not inplace:
            adata = adata.copy()
        assert all(u_entry in map.keys() for u_entry in adata.obs[column].unique()), \
            f'The map is missing entries! \nmap: {map.keys()}\nentries: {adata.obs[column].unique().tolist()}'
        
        adata.obs[f'{column}_cat'] = adata.obs.apply(lambda row: map[row[column]], axis=1).astype('category')
        if add_catcodes:
            adata.obs[f'{column}_catcode'] = adata.obs[f'{column}_cat'].cat.codes
        if not inplace:
            return adata
        
    def regress_out(self, adata: ad.AnnData, column: str, inplace=True) -> None | ad.AnnData:
        if not inplace:
            adata = adata.copy()
        adata = adata[adata.obs[column].astype('float32').notna()]
        assert not any(np.isnan(adata.obs[column].to_numpy())), 'Column {} still contains NaNs!'.format(column)
        assert not adata.obs[column].hasnans, 'Column {} still contains NaNs!'.format(column)
        sc.pp.regress_out(adata, column)
        
        if not inplace:
            return adata
        
    def combine_ct(self, 
                   adata: ad.AnnData, 
                   cell_types: Collection[str], 
                   combine_leftovers: Literal['all'] | Collection[str] | None, 
                   leftover_name: str='Other') -> ad.AnnData:
        result = adata.copy()
        _comb_left: int = 1 if combine_leftovers != None else 0
        columns = []
        X_new = []
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

    @staticmethod
    def _pca_screeplot(pca_model: PCA, figsize: tuple[int, int]=(8,8), title_add: str='') -> None:
        """
        Shows a barplot of the fraction of variance explained by each PC.
        """
        var_cor, var_contrib = Preprocessor._pca_summary(pca_model)
        dim = np.arange(var_cor.shape[1])
        var = pca_model.explained_variance_ratio_
        cumvar = np.cumsum(pca_model.explained_variance_ratio_)

        plt.figure(figsize=figsize)
        plt.rc('axes', axisbelow=True)
        plt.grid(axis = 'y', linewidth = 0.5)
        plt.bar(dim, 100*var).set_label('Individual Var.')
        plt.scatter(dim, cumvar*100, c='red').set_label('Cumulative Var.')
        plt.xticks(dim, dim+1, rotation=90)
        plt.xlabel('Component')
        plt.ylabel('Explained Variance (%)')
        plt.title(f'Variance Plot {title_add}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _pca_summary(pca_model: PCA) -> tuple[np.ndarray, np.ndarray]:
        sdev = np.sqrt(pca_model.explained_variance_)
        var_cor = np.apply_along_axis(lambda c: c*sdev, 0, pca_model.components_)
        var_cos2 = var_cor**2
        comp_cos2 = np.sum(var_cos2, 1)
        var_contrib = np.apply_along_axis(lambda c: c/comp_cos2, 0, var_cos2)
        return var_cor, var_contrib