from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

def kfold_cv(model_obj: object, X: pd.DataFrame, y: pd.Series, n_splits: int=5, return_df: bool=False) -> dict[str, dict] | pd.DataFrame:
    cv_results: dict[str, dict] = {n: {'acc': None, 'prc': None, 'rec': None, 'f1': None} for n in range(1, n_splits+1)}
    split = 1

    for train_i, test_i in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(X, y):
        X_train, X_test = X.iloc[train_i], X.iloc[test_i]
        y_train, y_test = y.iloc[train_i], y.iloc[test_i]
        
        print(f'Starting iteration {split}...')
        print(f'\tX_train.shape={X_train.shape}')
        print(f'\ty_train.shape={y_train.shape}')

        print('Training model... ', end='')
        m = model_obj.fit(X_train, y_train)

        print('testing model... ', end='')
        pred = m.predict(X_test)
        cv_results[split]['acc'] = accuracy_score(y_test, pred)
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
        cols=[i for i in range(1, n_splits+1)] + ['avg']
        cv_res_df = pd.DataFrame(data=cv_results, columns=cols)
        return cv_res_df
    else:
        return cv_results