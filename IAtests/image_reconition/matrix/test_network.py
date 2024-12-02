import numpy as np
import pandas as pd
from networkMatrix import NetworkMatrix
import matplotlib.pyplot as plt
import shelve


def get_dataset():
    iris = pd.read_csv('../../dataset/Iris.csv')

    iris.drop('Id', axis=1, inplace=True)
    species_map = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 1
    }
    iris['Species'] = iris['Species'].map(species_map)
    return iris

def dataset_to_training_data(iris):
    # X = iris[['SepalLengthCm', 'SepalWidthCm']].to_numpy()
    X = iris[['PetalLengthCm', 'PetalWidthCm']].to_numpy()
    # X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy()
    target = iris['Species'].to_numpy().reshape(-1, 1)
    return X, target


def create_model(X, y, IA, nb_iter = 100):
    X, y = X.T, y.T
    Loss, Accuracy = IA.train(X, y, nb_iter)
    
    plt.plot(np.array(Loss) / max(Loss))
    plt.plot(np.array(Accuracy))
    plt.show()
    IA.save()
    return IA

# iris = get_dataset()
# X, target = dataset_to_training_data(iris)

# create_model(X, target, width=5, length=3, learning_rate = 0.2, nb_iter=1000)
