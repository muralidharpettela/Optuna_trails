# import optuna
#
# def objective(trial):
#     x = trial.suggest_uniform('x', -10, 10)
#     return (x - 2) ** 2
#
# study = optuna.create_study()
# study.optimize(objective, n_trials=100)
#
# study.best_params  # E.g. {'x': 2.002108042}

import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection


def objective():
    iris = sklearn.datasets.load_iris()  # Prepare the data.

    clf = sklearn.ensemble.RandomForestClassifier(
        n_estimators=5, max_depth=3)  # Define the model.

    return sklearn.model_selection.cross_val_score(
        clf, iris.data, iris.target, n_jobs=-1, cv=3).mean()  # Train and evaluate the model.


print('Accuracy: {}'.format(objective()))

import optuna


def objective(trial):
    iris = sklearn.datasets.load_iris()

    n_estimators = trial.suggest_int('n_estimators', 2, 20)
    max_depth = int(trial.suggest_loguniform('max_depth', 1, 32))

    clf = sklearn.ensemble.RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth)

    return sklearn.model_selection.cross_val_score(
        clf, iris.data, iris.target, n_jobs=-1, cv=3).mean()


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

import sklearn.svm


def objective(trial):
    iris = sklearn.datasets.load_iris()

    classifier = trial.suggest_categorical('classifier', ['RandomForest', 'SVC'])

    if classifier == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 2, 20)
        max_depth = int(trial.suggest_loguniform('max_depth', 1, 32))

        clf = sklearn.ensemble.RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth)
    else:
        c = trial.suggest_loguniform('svc_c', 1e-10, 1e10)

        clf = sklearn.svm.SVC(C=c, gamma='auto')

    return sklearn.model_selection.cross_val_score(
        clf, iris.data, iris.target, n_jobs=-1, cv=3).mean()


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_slice(study)
optuna.visualization.plot_contour(study, params=['n_estimators', 'max_depth'])