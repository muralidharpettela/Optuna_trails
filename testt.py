from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist = fetch_openml(name='Fashion-MNIST', version=1)
classes = list(set(mnist.target))

# For demonstrational purpose, only use a subset of the dataset.
n_samples = 4000
data = mnist.data[:n_samples]
target = mnist.target[:n_samples]

x_train, x_test, y_train, y_test = train_test_split(data, target)