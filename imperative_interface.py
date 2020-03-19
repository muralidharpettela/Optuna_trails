'''
## Imperative Interface: Search Conditional Hyperparameters

Optuna deals with conditional hyperparameters with its imperative (define-by-run) interace.
Suppose that you are wondering which regularization method is better: `Ridge` or `Lasso`. You also want to optimize the regularization constant of each method.
In this case, you have three hyperparameters to be optimized.

- `regression_method`: `'ridge'` or `'lasso'`
- `ridge_alpha`: the regularization constant of `ridge`
- `lasso_alpha`: the regularization constant of `lasso`

Note that `ridge_alpha` and `lasso_alpha` are conditional hyperparameters:
`ridge_alpha` appears in the search space only when `regression_method` is `ridge`; and `lasso_alpha` does only when `regression_method` is `lasso`.
'''

import optuna
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics


def objective(trial):
    # hyperparameter setting
    regression_method = trial.suggest_categorical('regression_method', ('ridge', 'lasso'))
    if regression_method == 'ridge':
        ridge_alpha = trial.suggest_uniform('ridge_alpha', 0.0, 2.0)
        model = sklearn.linear_model.Ridge(alpha=ridge_alpha)
    else:
        lasso_alpha = trial.suggest_uniform('lasso_alpha', 0.0, 2.0)
        model = sklearn.linear_model.Lasso(alpha=lasso_alpha)

    # data loading and train-test split
    X, y = sklearn.datasets.load_boston(return_X_y=True)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state=0)

    # model training and evaluation
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    error = sklearn.metrics.mean_squared_error(y_val, y_pred)

    # output: evaluation score
    return error


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print('Minimum mean squared error: ' + str(study.best_value))
print('Best parameter: ' + str(study.best_params))

print(study.trials_dataframe())