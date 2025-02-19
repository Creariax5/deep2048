import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
import numpy as np

def convert_to_one_hot(y, nb_classes=3):
    y = y.ravel()
    one_hot = np.zeros((y.shape[0], nb_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    
    return one_hot

def convert_from_one_hot(one_hot):
    return np.argmax(one_hot, axis=1)

def get_dataset():
    X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
    X_circles, y_circles = make_circles(n_samples=200, noise=0.04, random_state=42)
    X = np.concatenate([X_moons, X_circles * 2, X_moons * 0.4 -2])
    y = np.concatenate([y_moons, y_circles + 1, y_moons])

    # print(X.shape, y.shape)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    return X, y

def split_data(X, y, train_percentage=0.9):
    # Generate random indices for splitting
    indices = np.random.permutation(len(X))
    split_index = int(train_percentage * len(X))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    # Split data using random indices
    train_data = X[train_indices]
    test_data = X[test_indices]
    train_labels = y[train_indices]
    test_labels = y[test_indices]
    
    return train_data, test_data, train_labels, test_labels
