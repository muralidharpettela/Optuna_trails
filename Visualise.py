# Preparing the Dataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist = fetch_openml(name='Fashion-MNIST', version=1)
classes = list(set(mnist.target))

# For demonstrational purpose, only use a subset of the dataset.
n_samples = 4000
data = mnist.data[:n_samples]
target = mnist.target[:n_samples]

x_train, x_test, y_train, y_test = train_test_split(data, target)

from sklearn.neural_network import MLPClassifier


def objective(trial):
    clf = MLPClassifier(
        hidden_layer_sizes=tuple([trial.suggest_int(f'n_units_l{i}', 32, 64) for i in range(3)]),
        learning_rate_init=trial.suggest_loguniform('lr_init', 1e-5, 1e-1),
    )

    for step in range(100):
        clf.partial_fit(x_train, y_train, classes=classes)
        value = clf.score(x_test, y_test)

        # Report intermediate objective value.
        trial.report(value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune(step):
            raise optuna.exceptions.TrialPruned()

    return value

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)  # This verbosity change is just to simplify the notebook output.

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)

from optuna.visualization import plot_optimization_history

plot_optimization_history(study)

