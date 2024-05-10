from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import pandas as pd
import numpy as np
from collections.abc import Callable
from typing import Any
from plotnine import ggplot, labs, aes, geom_point, geom_errorbar


def kfold_cv(model_obj: object, X: pd.DataFrame, y: pd.Series, k: int=5, return_df: bool=False, random_state: int=123) -> dict[str, dict] | pd.DataFrame:
    cv_results: dict[str, dict] = {n: {'acc': None, 'prc': None, 'rec': None, 'f1': None} for n in range(1, k+1)}
    split = 1

    for train_i, test_i in StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state).split(X, y):
        X_train, X_test = X.iloc[train_i], X.iloc[test_i]
        y_train, y_test = y.iloc[train_i], y.iloc[test_i]
        
        print(f'Starting iteration {split}...')
        print(f'\tX_train.shape={X_train.shape}')
        print(f'\ty_train.shape={y_train.shape}')

        print('Training model... ', end='')
        m = model_obj.fit(X_train, y_train)

        print('testing model... ', end='')
        pred = m.predict(X_test)
        cv_results[split]['acc'] = balanced_accuracy_score(y_test, pred)
        cv_results[split]['prc'] = precision_score(y_test, pred, average='macro')
        cv_results[split]['rec'] = recall_score(y_test, pred, average='macro')
        cv_results[split]['f1'] = f1_score(y_test, pred, average='macro')
        split += 1
        print('complete!\n')

    cv_results['avg'] = {'acc': None, 'prc': None, 'rec': None, 'f1': None}
    for metric in ['acc', 'prc', 'rec', 'f1']:
        _ = []
        for v in cv_results.values():
            if v[metric] is None:
                continue
            _.append(v[metric])
        cv_results['avg'][metric] = np.mean(_)

    if return_df:
        cols=[i for i in range(1, k+1)] + ['avg']
        cv_res_df = pd.DataFrame(data=cv_results, columns=cols)
        return cv_res_df
    else:
        return cv_results
    

Xt, Xv = pd.DataFrame | np.ndarray
yt, yv = pd.Series | np.ndarray

def optimize_hyperparam(X: pd.DataFrame,
                        y: pd.Series,
                        k: int, 
                        H: list, 
                        cv_func: Callable[[Xt, yt, Xv, yv, Any], pd.DataFrame], 
                        random_state: int=123) -> dict[str, dict] | pd.DataFrame:
    """
    Do stratified k-fold cross-validation with a dataset, to check how a model behaves as a function
    of the values in H (eg. a hyperparameter such as tree depth, or polynomial degree).

    :param X: feature matrix.
    :param y: response column.
    :param k: number of folds.
    :param H: values of the hyperparameter to cross-validate.
    :param cv_fun: function of the form (X_train, y_train, X_valid, y_valid, h) to evaluate the model in one split,
        as a function of h. It must return a dictionary with metric score values.
    :param random_state: controls the pseudo random number generation for splitting the data.
    :return: a Pandas dataframe with metric scores along values in H.
    """
    kf = StratifiedKFold(n_splits = k, shuffle = True, random_state = random_state)
    scores = []  # to store global results

    # for each value h in H, do CV
    for h in H:
        
        # for each fold 1..K
        kk = 0
        for train_index, valid_index in kf.split(X, y):
            kk = kk+1
            # partition the data in training and validation
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            # call cv_fun to train the model and compute performance
            fold_scores = {'_h': h, '_fold': kk}
            fold_scores.update(cv_func(X_train, y_train, X_valid, y_valid, h))
            scores.append(fold_scores)
            
    return pd.DataFrame(scores)


def plot_optimization_results(results: pd.DataFrame, h_var: str='_h', fold_var: str='fold', var_name: str='metric') -> None:
    res_melt = results.melt(id_vars=[h_var, fold_var], var_name=var_name)
    res_melt = res_melt.groupby(['_h', 'metric'], as_index=False).aggregate(stdev = ('value', 'std'), mean = ('value', 'mean'))

    plot = ggplot(res_melt, aes('_h', 'mean', color = 'metric')) +\
    geom_point() +\
    geom_errorbar(aes(ymin='mean-stdev', ymax='mean+stdev'), width=.02) +\
    labs(x='Threshold', y='F1-score', title='Macro-averaged F-score for Different Thresholds')
    plot.draw(True)
