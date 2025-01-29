import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
import numpy as np

def get_dataset():
    X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
    X_circles, y_circles = make_circles(n_samples=200, noise=0.04, random_state=42)
    X = np.concatenate([X_moons, X_circles * 2, X_moons * 0.4 -2])
    y = np.concatenate([y_moons, y_circles, y_moons])
    return X, y

X, y = get_dataset()
print(X.shape, y.shape)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
