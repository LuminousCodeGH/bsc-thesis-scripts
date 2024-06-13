import pandas as pd
import numpy as np
import anndata as ad
import os.path

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.base import BaseEstimator
from core.normalizer import Normalizer
from collections.abc import Iterable, Callable
from typing import Any, Literal
from plotnine import ggplot, aes, geom_boxplot, labs, geom_point, geom_errorbar, theme, element_text, geom_bar, position_dodge, stat_summary


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
        self.best_normalization: int = None
        self.gridsearch_results: dict[str, pd.DataFrame] = {}
        self.gridsearch_models: dict[str: BaseEstimator] = {}

    @property
    def results(self) -> pd.DataFrame:
        if not hasattr(self, '_results'):
            raise AttributeError('No results are available. Test models first!')
        return self._results
    
    @property
    def gridsearch_results_summary(self) -> dict[str, pd.DataFrame]:
        if not hasattr(self, '_gridsearch_results_summary'):
            _gs_sum: dict[str, pd.DataFrame] = {}
            for name, df in self.gridsearch_results.items():
                _df = df.iloc(axis=1)[(df.columns.str.endswith('test_score') | 
                                df.columns.str.endswith('train_score') | 
                                df.columns.str.startswith('param')) & 
                                (~df.columns.str.startswith('split')) &
                                (~df.columns.str.startswith('rank'))]
                _gs_sum[name] = _df
            self._gridsearch_results_summary = _gs_sum
        return self._gridsearch_results_summary

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
            if len(self.categories) == 2:
                scores_dict = {'acc' : balanced_accuracy_score(y_test, pred),
                            'f1': f1_score(y_test, pred, average='macro'),
                            'auc': roc_auc_score(y_test, pred)}
            elif len(self.categories) > 2:
                scores_dict = {'acc' : balanced_accuracy_score(y_test, pred),
                            'f1': f1_score(y_test, pred, average='macro')}
            fold_scores.update(scores_dict)
            split += 1
            if self.verbose:
                print('complete!\n')
            cv_results.append(fold_scores)

        return pd.DataFrame(cv_results)
    
    def _get_X_and_y(self, adata: ad.AnnData, idx: int | Literal['best'] | None=None, ignore_normalization: bool=False) -> tuple[np.ndarray, np.ndarray]:
        if self.normalizer is not None and not ignore_normalization:
            if idx == 'best':
                idx = self.best_normalization
            if idx is None:
                print('WARNING: Normalizer is set but index is unset. Choosing the first normalization!')
                _layer = self.normalizer.layer_names[0]
                if _layer not in adata.layers.keys():
                    adata = self.normalizer.normalize_all(adata, False)
            elif idx is not None:
                if self.verbose:
                    print(f'Normalizing index {idx}... (={self.normalizer.normalizations[idx]})')
                _layer = self.normalizer.layer_names[idx]
                if _layer not in adata.layers.keys():
                    adata = self.normalizer._normalize(adata, idx, False)
            X: np.ndarray = adata.layers[_layer][adata.obs[self.metric_column].isin(self.categories)]
        else:
            X: np.ndarray = adata.X[adata.obs[self.metric_column].isin(self.categories)]

        y: np.ndarray = adata.obs[self.metric_column][adata.obs[self.metric_column].isin(self.categories)].to_numpy()
        if self.verbose:
            print(f'Getting X and y: X={X.shape}, y={y.shape}')
        return X, y
    
    def _plot_results(self, title: str, average_stat: Literal['mean', 'median'], plot_type: Literal['box', 'bar'], final_test: bool) -> None:
        if not final_test:
            res_melt = self.results[['_h', '_fold', 'f1', 'auc', 'acc']].melt(id_vars=['_h', '_fold'], var_name='metric')
        else:
            if not hasattr(self, 'final_results'):
                raise AttributeError('The final test has not been performed yet! (Final test results not found)')
            res_melt = self.final_results[['_h', '_fold', 'f1', 'auc', 'acc']].melt(id_vars=['_h', '_fold'], var_name='metric')

        if plot_type == 'box':
            plot = ggplot(res_melt, aes('_h', 'value', fill='metric')) +\
                geom_boxplot(notch=False, mapping=aes(middle=f'np.{average_stat}(value)')) +\
                labs(x='Classifier', y='Score', title=title)
            plot.draw(True)

        elif plot_type == 'bar':
            res_melt = res_melt.groupby(['_h', 'metric'], as_index=False).aggregate(
                stdev = ('value', 'std'), 
                mean = ('value', 'mean'), 
                median = ('value', 'median'))
            plot = ggplot(res_melt, aes('_h', 'median', fill='metric')) +\
                geom_bar(stat='identity', position='dodge') +\
                geom_errorbar(aes(ymin='mean-stdev', ymax='mean+stdev', group='metric'), position=position_dodge(0.9), width=.02) +\
                labs(x='Classifier', y='Score', title=title) +\
                stat_summary(mapping=aes('_h', 'mean'), fun_data='mean_cl_boot', position=position_dodge(0.9))
            plot.draw(True)
    
    def test_model(self, 
                   adata: ad.AnnData, 
                   model: BaseEstimator, 
                   name: str,
                   is_baseline: bool=False) -> pd.DataFrame:
        X, y = self._get_X_and_y(adata, ignore_normalization=is_baseline)
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
        _res = self.test_model(adata, self.baseline_model, name, True)
        self._results = _res
        return _res
    
    def test_normalizations(self,
                            adata: ad.AnnData,
                            names: Iterable[str]=None) -> pd.DataFrame:
        if names is not None:
            assert len(names) == len(self.normalizer.normalizations), 'The length of the names and the number of different normalizations do not match!'

        model = self.baseline_model
        _results: list[pd.DataFrame] = []
        _best_normalization_score = 0
        _best_normalization: str = None
        for i in range(len(self.normalizer.layer_names)):
            name = self.normalizer.layer_names[i]
            if names is not None:
                name = names[i]
            X, y = self._get_X_and_y(adata, i)
            model.random_state = self.random_state
            _results.append(self._kfold_cv(model, X, y).assign(_h=name))
            _acc = np.mean(_results[-1]['acc'])
            _best_normalization = i if _acc > _best_normalization_score else _best_normalization
        self.best_normalization = i

        _res = pd.concat(_results, axis=0)
        if hasattr(self, '_results'):
            self._results = pd.concat([self._results, _res], axis=0)
        return _res
        
    def plot_results(self, title: str, average_stat: Literal['mean', 'median']='median', plot_type: Literal['box', 'bar'] = 'bar') -> None:
        self._plot_results(title, average_stat, plot_type, False)

    def plot_final_results(self, title: str, average_stat: Literal['mean', 'median']='median', plot_type: Literal['box', 'bar'] = 'bar') -> None:
        self._plot_results(title, average_stat, plot_type, True)
    
    def exhaustive_gridsearch(self, 
                              adata: ad.AnnData, 
                              model: BaseEstimator,
                              model_name: str, 
                              grid_parameters: dict[str: Any],
                              scoring: str | Callable[[np.ndarray, np.ndarray], float] = make_scorer(balanced_accuracy_score),
                              normalization_idx: int | Literal['best'] | None=None,
                              save_type: Literal['full', 'summ'] = None,
                              file_name: str=None,
                              override_file: bool=False) -> None:
        self.gridsearch_models[model_name] = model

        if file_name is not None:
            if not override_file and os.path.isfile(file_name):
                print('Retrieving previous results...', end=' ')
                self.import_gridsearch_results(file_name, model_name)
                print('done! Stopping here...')
                return
        
        model.random_state = self.random_state
        X, y = self._get_X_and_y(adata, normalization_idx)
        print('Fitting GridSearchCV...', end=' ')
        grid_cv = GridSearchCV(model, 
                               param_grid=grid_parameters, 
                               cv=self.k, 
                               scoring=scoring, 
                               n_jobs=-1,
                               return_train_score=True,
                               refit=False).fit(X, y)
        print('done!')

        _df: pd.DataFrame = pd.DataFrame(grid_cv.cv_results_)
        _df['median_test_score'] = _df.filter([f'split{i}_test_score' for i in range(self.k)]).median(axis=1)
        _df['median_train_score'] = _df.filter([f'split{i}_train_score' for i in range(self.k)]).median(axis=1)

        if save_type == 'summ':
            _df = _df.iloc(axis=1)[(_df.columns.str.endswith('test_score') | 
                                    _df.columns.str.endswith('train_score') | 
                                    _df.columns.str.startswith('param')) & 
                                    (~_df.columns.str.startswith('split')) &
                                    (~_df.columns.str.startswith('rank'))]
        elif save_type == 'full':
            pass

        self.gridsearch_results[model_name] = _df

        if file_name is not None:
            _df.to_csv(file_name)

    def import_gridsearch_results(self, file_name: str, model_name: str) -> None:
        self.gridsearch_results[model_name] = pd.read_csv(file_name, index_col=0)

    def run_final_test(self, train_adata: ad.AnnData, test_adata: ad.AnnData, random_states: Iterable[int], normalization: str) -> pd.DataFrame:
        best_models: dict[str, BaseEstimator] = {}

        # For every gridsearched model get the best parameters and apply them
        from json import loads
        for model_name, df in self.gridsearch_results.items():
            base_model: BaseEstimator = self.gridsearch_models[model_name]
            _best_params_str = df.sort_values(['median_test_score', 'mean_test_score'], ascending=[False, False]).params[0].replace("'", '"')
            best_params = loads(_best_params_str)
            best_model = base_model.set_params(**best_params)
            best_models[model_name] = best_model
        
        X_train = train_adata.X if normalization is None else train_adata.layers[f'norm_{normalization}']
        y_train = train_adata.obs[self.metric_column][train_adata.obs[self.metric_column].isin(self.categories)].to_numpy()

        X_test = test_adata.X if normalization is None else test_adata.layers[f'norm_{normalization}']
        y_test = test_adata.obs[self.metric_column][test_adata.obs[self.metric_column].isin(self.categories)].to_numpy()

        results: list[pd.DataFrame] = []

        # Do a k-fold CV type thing only randomizing the models instead
        _iter_scores: list[dict] = []
        _iter_score: dict = {}
        for model_name, model in best_models.items():
            _iter = 0
            for random_state in random_states:
                model.random_state = random_state
                m = model.fit(X_train, y_train)
                pred = m.predict(X_test)

                _iter_score = {'_fold': _iter, 'model': m}
                if len(self.categories) == 2:
                    scores_dict = {'acc' : balanced_accuracy_score(y_test, pred),
                                'f1': f1_score(y_test, pred, average='macro'),
                                'auc': roc_auc_score(y_test, pred)}
                elif len(self.categories) > 2:
                    scores_dict = {'acc' : balanced_accuracy_score(y_test, pred),
                                'f1': f1_score(y_test, pred, average='macro')}
                _iter_score.update(scores_dict)
                _iter += 1
                _iter_scores.append(_iter_score)
            results.append(pd.DataFrame(_iter_scores).assign(_h=model_name))
        self.final_results = pd.concat(results, axis=0)
        return self.final_results

    def plot_overfit(self, name_in_gs_results: str, param_on_x: str, average_stat: Literal['mean', 'median'], title: str):
        test_df = self.gridsearch_results[name_in_gs_results].melt(id_vars=[param_on_x, 'std_test_score'], value_vars=[f'{average_stat}_test_score'])
        train_df = self.gridsearch_results[name_in_gs_results].melt(id_vars=[param_on_x, 'std_train_score'], value_vars=[f'{average_stat}_train_score'])

        plot = ggplot() +\
            geom_point(data=test_df, mapping=aes(x=param_on_x, y='value', color='variable')) +\
            geom_point(data=train_df, mapping=aes(x=param_on_x, y='value', color='variable')) +\
            labs(x=' '.join(param_on_x.split('_')[1:]).capitalize(), y='Score', title=title) +\
            theme(axis_text_x=element_text(rotation=45))
        if average_stat == 'mean':
            plot = plot +\
                geom_errorbar(data=test_df, mapping=aes(x=param_on_x, ymin='value-std_test_score', ymax='value+std_test_score')) +\
                geom_errorbar(data=train_df, mapping=aes(x=param_on_x, ymin='value-std_train_score', ymax='value+std_train_score'))
        plot.draw(True)

    @staticmethod
    def split_test_data(adata: ad.AnnData, test_samples: float | int, stratify_on: str, random_state: int=None) -> tuple[ad.AnnData, ad.AnnData]:
        '''Splits the adata according to a percentage if test_samples is a float, or according to an absolute number of samples. 
        Returns train_adata, test_adata in this order'''
        total_samples: int = adata.shape[0]
        if isinstance(test_samples, int):
            print(f'{test_samples} samples = {(test_samples / total_samples) * 100:.3f}%')
            test_samples = round(test_samples / total_samples, 3)
        train_idx, test_idx = train_test_split(range(adata.shape[0]), test_size=test_samples, random_state=random_state, stratify=adata.obs[stratify_on])
        train_adata = adata[train_idx, :]
        test_adata = adata[test_idx, :]
        print(f'{train_adata.shape=}')
        print(f'{test_adata.shape=}')

        return train_adata, test_adata
