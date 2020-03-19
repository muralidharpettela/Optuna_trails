## Optimize Machine Learning Models

##Let's optimize the following machine learning logic, where a linear regression model (Lasso) is trained for the Boston Housing dataset.
# Original model
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics

# hyperparameter setting
alpha = 1.0

# data loading and train-test split
X, y = sklearn.datasets.load_boston(return_X_y=True)
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state=0)

# model training and evaluation
model = sklearn.linear_model.Lasso(alpha=alpha)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
error = sklearn.metrics.mean_squared_error(y_val, y_pred)

# output: evaluation score
print('Mean squared error: ' + str(error))


# Optimizing the model

import optuna
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics


def objective(trial):
    # hyperparameter setting
    alpha = trial.suggest_uniform('alpha', 0.0, 2.0)

    # data loading and train-test split
    X, y = sklearn.datasets.load_boston(return_X_y=True)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state=0)

    # model training and evaluation
    model = sklearn.linear_model.Lasso(alpha=alpha)
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