import scanpy as sc
import anndata as ad
import numpy as np
from sklearn.preprocessing import normalize, minmax_scale, robust_scale
from conorm import tmm, mrn
from collections.abc import Callable, Collection
from typing import Literal, Any


class Normalizer:
    def __init__(self, 
                 normalizations: Collection[Literal['l1', 'l2', 'minmax', 'robust', 'tmm', 'mrn', 'ref [CT]']], 
                 transformation: Callable[[np.ndarray], None],
                 scale_factor: int = 10000,
                 keyword_args: dict[Literal['l1', 'l2', 'minmax', 'robust', 'tmm', 'mrn', 'ref [CT]'], dict[str, Any]] = None) -> None:
        self.normalizations = list(normalizations)
        self.transformation = transformation
        self.scale_factor = scale_factor
        self.keyword_args = keyword_args
        if keyword_args is None:
            self.keyword_args = dict()
        
    @property
    def normalizations(self) -> list[Literal['l1', 'l2', 'minmax', 'robust', 'tmm', 'mrn', 'ref [CT]']]:
        assert len(self._normalizations) == len(self.layer_names), 'Length of normalizations and layers does not match!'
        return self._normalizations
    
    @normalizations.setter
    def normalizations(self, value: Collection[Literal['l1', 'l2', 'minmax', 'robust', 'tmm', 'mrn', 'ref [CT]']]) -> None:
        self._normalizations = value
        self.layer_names: list[str] = [self._generate_name_from_normalization(n) for n in value]

    def normalize_all(self, adata: ad.AnnData, inplace: bool=True) -> None | ad.AnnData:
        if not inplace:
            adata = adata.copy()

        for idx in range(len(self.normalizations)):
            self._normalize(adata, idx, inplace)

        if not inplace:
            return adata

    def _generate_name_from_normalization(self, index: int | str) -> str:
        _name: str
        if isinstance(index, int):
            _name = self._normalizations[index]
        elif isinstance(index, str):
            assert index in self._normalizations, 'The index provided ({}) is not in the normalizations!'.format(index)
            _name = index
        return f'norm_{_name}'

    def _normalize(self, adata: ad.AnnData, idx: int, inplace: bool=True) -> None | ad.AnnData:
        if not inplace:
            adata = adata.copy()

        _layer_name: str = self.layer_names[idx]
        _kwargs: dict[str, Any] = self.keyword_args.get(self.normalizations[idx], {})
        
        adata.layers[_layer_name] = adata.X.copy()
        
        match self.normalizations[idx].split(' ')[0]:
            case 'l1':
                sc.pp.normalize_total(adata, layer=_layer_name, target_sum=self.scale_factor)
            case 'l2':
                adata.layers[_layer_name] = normalize(adata.layers[_layer_name], 'l2', axis=1) * self.scale_factor
            case 'minmax':
                adata.layers[_layer_name] = minmax_scale(adata.layers[_layer_name], axis=1) * self.scale_factor
            case 'robust':
                adata.layers[_layer_name] = robust_scale(adata.layers[_layer_name], axis=1, with_centering=False, **_kwargs) * self.scale_factor
            case 'tmm':
                adata.layers[_layer_name] = tmm(adata.layers[_layer_name].T, **_kwargs).T
            case 'mrn':
                adata.layers[_layer_name] = mrn(adata.layers[_layer_name].T).T
            case 'ref':
                _ref_ct: str = ' '.join(self.normalizations[idx].split(' ')[1:])
                assert _ref_ct in adata.var_names.to_list(), 'Reference cell type {} not in adata!'.format(_ref_ct)
                _ref_idx = adata.var_names.get_loc(_ref_ct)        
                for row in adata.layers[_layer_name]:
                    row = row / row[_ref_idx]
        
        if self.transformation is not None:
            adata.layers[_layer_name] = self.transformation(adata.layers[_layer_name])

        assert np.nan not in adata.layers[_layer_name].flatten(), 'NaN in normalized layer {}!'.format(_layer_name)

        if not inplace:
            return adata
        
    def __str__(self) -> None:
        return '/n'.join([f'{norm} - {layer}' for norm, layer in zip(self.normalizations, self.layer_names)])
        