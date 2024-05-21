import pandas as pd
import numpy as np
import anndata as ad

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.base import BaseEstimator
from core.normalizer import Normalizer
from collections.abc import Iterable
from plotnine import ggplot, aes, geom_boxplot, labs


class ModelTester:
    def __init__(self, 
                 normalizer: Normalizer | None, 
                 baseline_model: BaseEstimator,
                 metric_column: str, 
                 categories: Iterable[str | float],
                 k: int=10,
                 random_state: int | None=None,
                 verbose: bool=False) -> None:
        self.normalizer = normalizer
        self.baseline_model = baseline_model
        self.metric_column = metric_column
        self.categories = categories
        self.k = k
        self.random_state = random_state
        self.verbose = verbose

    @property
    def results(self) -> pd.DataFrame:
        if not hasattr(self, '_results'):
            raise AttributeError('No results are available. Test models first!')
        return self._results

    def _kfold_cv(self,
                  model: BaseEstimator, 
                  X: np.ndarray, 
                  y: np.ndarray) -> pd.DataFrame:
        if self.verbose:
            print(f'Initializing kfold cv -> X={X.shape}, y={y.shape}')
        X = pd.DataFrame(X)
        y = pd.Series(y)
        cv_results: list[dict] = []
        k = self.k
        split = 0

        for train_i, test_i in StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state).split(X, y):
            X_train, X_test = X.iloc[train_i], X.iloc[test_i]
            y_train, y_test = y.iloc[train_i], y.iloc[test_i]
            
            if self.verbose:
                print(f'Starting iteration {split}...')
                print(f'\tX_train.shape={X_train.shape}')
                print(f'\ty_train.shape={y_train.shape}')

                print('Training model... ', end='')
            m = model.fit(X_train, y_train)

            if self.verbose:
                print('testing model... ', end='')
            pred = m.predict(X_test)

            if isinstance(pred[0], float):
                for i in range(len(pred)):
                    pred[i] = min(self.categories, key=lambda x: abs(x-pred[i]))  # In case of a regression, set the value to the closest category

            fold_scores = {'_fold': split}
            scores_dict = {'acc' : balanced_accuracy_score(y_test, pred),
                           'f1': f1_score(y_test, pred, average='macro')}
            fold_scores.update(scores_dict)
            split += 1
            if self.verbose:
                print('complete!\n')
            cv_results.append(fold_scores)

        return pd.DataFrame(cv_results)
    
    def _get_X_and_y(self, adata: ad.AnnData) -> tuple[np.ndarray, np.ndarray]:
        if self.normalizer is not None:
            if len(self.normalizer.layer_names) > 1:
                print('WARNING: Normalizer has more than one normalization. Choosing the first!')
            _layer = self.normalizer.layer_names[0]
            adata = self.normalizer.normalize_all(adata, False)
            X: np.ndarray = adata.layers[_layer][adata.obs[self.metric_column].isin(self.categories)]
        else:
            X: np.ndarray = adata.X[adata.obs[self.metric_column].isin(self.categories)]

        y: np.ndarray = adata.obs[self.metric_column][adata.obs[self.metric_column].isin(self.categories)].to_numpy()
        if self.verbose:
            print(f'Getting X and y: X={X.shape}, y={y.shape}')
        return X, y
    
    def test_model(self, 
                   adata: ad.AnnData, 
                   model: BaseEstimator, 
                   name: str) -> pd.DataFrame:
        X, y = self._get_X_and_y(adata)
        assert X.shape[0] == y.shape[0], 'X and y shapes do not match!'
        model.random_state = self.random_state
        _res = self._kfold_cv(model, X, y).assign(_h=name)

        if hasattr(self, '_results'):
            self._results = pd.concat([self._results, _res], axis=0)
        return _res

    def test_models(self, 
                   adata: ad.AnnData, 
                   models: Iterable[BaseEstimator], 
                   names: Iterable[str]) -> pd.DataFrame:
        if self.verbose:
            print('Testing model...')
        X, y = self._get_X_and_y(adata)
        assert X.shape[0] == y.shape[0], 'X and y shapes do not match!'
        assert len(models) == len(names), 'The number of models does not match the number of column names!'

        _results: list[pd.DataFrame] = []
        for name, model in zip(names, models):
            model.random_state = self.random_state
            _results.append(self._kfold_cv(model, X, y).assign(_h=name))
        
        _res = pd.concat(_results, axis=0)
        if hasattr(self, '_results'):
            self._results = pd.concat([self._results, _res], axis=0)
        return _res
    
    def test_baseline(self, 
                      adata: ad.AnnData, 
                      name: str) -> pd.DataFrame:
        _res = self.test_model(adata, self.baseline_model, name)
        self._results = _res
        return _res
    
    def plot_results(self, title: str) -> None:
        res_melt = self.results.melt(id_vars=['_h', '_fold'], var_name='metric')

        plot = ggplot(res_melt, aes('_h', 'value', fill='metric')) +\
        geom_boxplot(notch=False) +\
        labs(x='Classifier', y='Score', title=title)
        plot.draw(True)
    