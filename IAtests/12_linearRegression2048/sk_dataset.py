import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
import numpy as np

def get_dataset():
    X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)
    X_circles, y_circles = make_circles(n_samples=200, noise=0.04, random_state=42)
    # X = np.concatenate([X_moons, X_circles * 2 , X_moons * 0.4 -2]).reshape(-1, 1)
    # y = np.concatenate([y_moons, y_circles, y_moons]).reshape(-1, 1)
    # X = (X_circles[:, 0]).reshape(-1, 1)
    # y = (X_circles[:, 1]).reshape(-1, 1)
    X = (X_moons[:, 0]).reshape(-1, 1)
    y = (X_moons[:, 1]).reshape(-1, 1)

    print(X.shape, y.shape)
    plt.scatter(X, y)
    plt.show()
    return X, y
