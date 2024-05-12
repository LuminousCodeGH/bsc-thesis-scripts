from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import pandas as pd
import numpy as np
from collections.abc import Callable
from typing import Any, Literal
from plotnine import ggplot, labs, aes, geom_point, geom_errorbar, geom_boxplot


def kfold_cv(model_obj: object, X: pd.DataFrame, y: pd.Series, k: int=5, return_df: bool=False, random_state: int=123, verbose: bool=False) -> pd.DataFrame:
    cv_results: list[dict] = []
    split = 0

    for train_i, test_i in StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state).split(X, y):
        X_train, X_test = X.iloc[train_i], X.iloc[test_i]
        y_train, y_test = y.iloc[train_i], y.iloc[test_i]
        
        if verbose:
            print(f'Starting iteration {split}...')
            print(f'\tX_train.shape={X_train.shape}')
            print(f'\ty_train.shape={y_train.shape}')

            print('Training model... ', end='')
        m = model_obj.fit(X_train, y_train)

        if verbose:
            print('testing model... ', end='')
        pred = m.predict(X_test)
        fold_scores = {'_fold': split}
        scores_dict = {'acc' : balanced_accuracy_score(y_test, pred),
                       'f1': f1_score(y_test, pred, average='macro')}
        fold_scores.update(scores_dict)
        split += 1
        if verbose:
            print('complete!\n')
        cv_results.append(fold_scores)

    return pd.DataFrame(cv_results)
    

Xt = pd.DataFrame | np.ndarray
Xv = pd.DataFrame | np.ndarray
yt = pd.Series | np.ndarray
yv = pd.Series | np.ndarray

def optimize_hyperparam(X: pd.DataFrame,
                        y: pd.Series,
                        k: int, 
                        H: list, 
                        cv_func: Callable[[Xt, yt, Xv, yv, Any], dict[str, float]], 
                        H_mask: list[str] = None,
                        use_index: bool = False,
                        random_state: int=123) -> pd.DataFrame:
    """
    Do stratified k-fold cross-validation with a dataset, to check how a model behaves as a function
    of the values in H (eg. a hyperparameter such as tree depth, or polynomial degree).

    :param X: feature matrix.
    :param y: response column.
    :param k: number of folds.
    :param H: values of the hyperparameter to cross-validate.
    :param cv_func: function of the form (X_train, y_train, X_valid, y_valid, h) to evaluate the model in one split,
        as a function of h. It must return a dictionary with metric score values.
    :param H_mask: list containing subsitute names for the hyperparameters in H, use if you get long complex names as you _h.
    :param use_index: use the split index instead of passing the split dataframes
    :param random_state: controls the pseudo random number generation for splitting the data.
    :return: a Pandas dataframe with metric scores along values in H.
    """
    assert len(H_mask) == len(H) or H_mask is None, 'The H_mask must have the same length as H!'
    kf = StratifiedKFold(n_splits = k, shuffle = True, random_state = random_state)
    scores = []  # to store global results

    # for each value h in H, do CV
    for i, h in enumerate(H):
        print(f'Starting iteration h={h}', end='')
        # for each fold 1..K
        kk = 0
        for train_index, valid_index in kf.split(X, y):
            print('.', end='')
            kk = kk+1
            # partition the data in training and validation
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            # call cv_fun to train the model and compute performance
            h_name = h if H_mask is None else H_mask[i]
            fold_scores = {'_h': h_name, '_fold': kk}
            if not use_index:
                fold_scores.update(cv_func(X_train, y_train, X_valid, y_valid, h))
            else:
                fold_scores.update(cv_func(X_train, y_train, X_valid, y_valid, h))
            scores.append(fold_scores)
        print(' Done!')
            
    return pd.DataFrame(scores)


def plot_optimization_scatter(results: pd.DataFrame, title: str, h_var: str='_h', fold_var: str='_fold', var_name: str='metric') -> None:
    res_melt = results.melt(id_vars=[h_var, fold_var], var_name=var_name)
    res_melt = res_melt.groupby([h_var, var_name], as_index=False).aggregate(stdev = ('value', 'std'), mean = ('value', 'mean'))

    plot = ggplot(res_melt, aes(h_var, 'mean', color = var_name)) +\
    geom_point() +\
    geom_errorbar(aes(ymin='mean-stdev', ymax='mean+stdev'), width=.02) +\
    labs(x='Threshold', y='Score', title=title)
    plot.draw(True)


def plot_optimization_box(results: pd.DataFrame, title: str, h_var: str='_h', fold_var: str='_fold', var_name: str='metric') -> None:
    res_melt = results.melt(id_vars=[h_var, fold_var], var_name=var_name)

    plot = ggplot(res_melt, aes(h_var, 'value', fill = var_name)) +\
    geom_boxplot(notch=False) +\
    labs(x='Threshold', y='Score', title=title)
    plot.draw(True)
