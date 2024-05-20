import numpy as np
import anndata as ad
import pandas as pd

from typing import Literal
from collections.abc import Callable
from core.normalizer import Normalizer
from matplotlib.pyplot import Figure

class Differentiator:
    def __init__(self, 
                 normalization: Literal['l1', 'l2', 'minmax', 'robust', 'tmm', 'mrn', 'ref [CT]'],
                 test: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray[float], np.ndarray[float]]],
                 correction: Callable[[np.ndarray], tuple[np.ndarray[bool], np.ndarray[float]]],
                 metric_column: str,
                 categories: tuple[str | float, str | float],
                 alpha: float=0.05) -> None:
        self.normalizer: Normalizer = Normalizer(normalization, None)
        self.test = test
        self.correction = correction
        self.metric_column = metric_column
        self.categories = categories
        self.alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha
    
    @alpha.setter
    def alpha(self, value: float) -> None:
        if not 0.000 < value < 1.000:
            raise ValueError(f'Alpha must be between zero and one, not {value:.3f}!')
        self._alpha = value

    @property
    def abundance_df(self) -> pd.DataFrame | None:
        if not hasattr(self, '_abundance_df'):
            raise AttributeError('Abundance dataframe unset! Run Differentiator.differentiate first!')
        return self._abundance_df

    def differentiate(self, adata: ad.AnnData, inplace: bool=True) -> None:
        if not inplace:
            adata = adata.copy()
        column = self.metric_column
        _data = self.normalizer.normalize_all(adata, False)
        _norm_layer = self.normalizer.layer_names[0]
        X: np.ndarray = _data.layers[_norm_layer][_data.obs[column] == self.categories[0]]
        Y: np.ndarray = _data.layers[_norm_layer][_data.obs[column] == self.categories[1]]
        del _data

        results = self.test(X, Y, 'two-sided')
        corr_results = self.correction(results.pvalue, self.alpha)
        adata.var['significant'] = corr_results[0]
        adata.var['corr_pvalue'] = corr_results[1]

        if not hasattr(self, '_abundance_df'):
            self._abundance_df = pd.DataFrame(data=adata.var[['significant', 'corr_pvalue']], index=adata.var_names)
        else:
            self.abundance_df['significant'] = adata.var['significant'].copy()
            self.abundance_df['corr_pvalue'] = adata.var['corr_pvalue'].copy()

        if not inplace:
            return adata
        
    def volcano_plot(self, adata: ad.AnnData, pvalue_column: str='corr_pvalue', significant_column: str='significant') -> None:
        norm_l1 = Normalizer(['l1'], None)
        _norm_adata = norm_l1.normalize_all(adata, False)
        column = self.metric_column
        X: np.ndarray = _norm_adata.layers['norm_l1'][_norm_adata.obs[column] == self.categories[0]]
        Y: np.ndarray = _norm_adata.layers['norm_l1'][_norm_adata.obs[column] == self.categories[1]]
        del _norm_adata

        fold_changes = np.empty(adata.X.shape[1])
        print(f'log2( fold change ) = log2( {self.categories[0]}/{self.categories[1]} )')
        for i in range(adata.X.shape[1]):
            fold_changes[i] = np.log2(np.mean(X[:, i]) / np.mean(Y[:, i]))

        adata.var['log2fc'] = fold_changes
        adata.var['log10p'] = np.log10(adata.var[pvalue_column].to_numpy())*-1
        _volc_df = adata.var.reset_index()

        self.abundance_df['log2fc'] = adata.var['log2fc'].copy()
        self.abundance_df['log10p'] = adata.var['log10p'].copy()

        from plotnine import ggplot, aes, geom_point, geom_hline, geom_text, geom_vline, xlab, ylab, xlim, ggtitle

        volcano_plot = ggplot(_volc_df, aes('log2fc', 'log10p')) +\
            geom_point(alpha=0.5) +\
            geom_hline(yintercept=np.log10(self.alpha)*-1, color='black') +\
            geom_text(aes(label='index'), data=_volc_df[_volc_df[significant_column] == True], color='black', nudge_y=0.15, size=6) +\
            geom_vline(xintercept=(0.5, -0.5), color='black', linetype='dotted') +\
            xlab('log2( fold change )') + ylab('-log10( p-value )') +\
            xlim(-1.5, 1.5) +\
            ggtitle('Abundance Volcano Plot')
        volcano_plot.draw(True)
        del _volc_df


