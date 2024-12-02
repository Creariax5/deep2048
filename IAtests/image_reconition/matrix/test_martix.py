import numpy as np
import pandas as pd
from IAtests.image_reconition.matrix.perceptronMatrix import PerceptronMatrix
import matplotlib.pyplot as plt
import shelve

# Reading the data
iris = pd.read_csv('../../dataset/Iris.csv')

# Dropping the id column
iris.drop('Id', axis=1, inplace=True)
# iris.drop('SepalLengthCm', axis=1, inplace=True)
# iris.drop('SepalWidthCm', axis=1, inplace=True)

species_map = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 1
}

# Replace values using map
iris['Species'] = iris['Species'].map(species_map)

# Convert features (PetalLengthCm and PetalWidthCm) to numpy array
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy()

# Convert target variable (Species) to numpy array
target = iris['Species'].to_numpy()


def create_model(X, y, learning_rate = 0.1, nb_iter = 100):
    IA = PerceptronMatrix(X.shape[1], learning_rate)
    Loss, Accuracy = IA.train(X, y, nb_iter)
    
    plt.plot(np.array(Loss) / max(Loss))
    plt.plot(np.array(Accuracy))
    plt.show()
    IA.save()
    return IA

# create_model(X, target, 0.05, 10)

# IA = PerceptronMatrix(X.shape[1])
# IA.load()
# A = IA.forward_propagation(X)
# print(IA.log_loss(A, target))
