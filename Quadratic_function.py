# Optimize Quadratic Function
#Before optimizing a machine learning model, let's see how Optuna solves a very simple task that minimizes the output of $f(x) = (x - 2)^2$.
#Although the answer is obviously $f(x) = 0$ when $x = 2$, Let see how Optuna solve that.

import optuna

def objective(trial):
    x = trial.suggest_uniform('x', -100, 100)
    return (x - 2) ** 2

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print('Minimum objective value: ' + str(study.best_value))
print('Best parameter: ' + str(study.best_params))