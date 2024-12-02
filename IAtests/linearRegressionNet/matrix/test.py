import time

from matplotlib import pyplot as plt
import numpy as np
from networkMatrix import NetworkMatrix
import pil
import sk_dataset

def create_model(X, y, IA, nb_iter = 100):
    X, y = X.T, y.T
    Loss, Accuracy = IA.train(X, y, nb_iter)
    
    plt.plot(np.array(Loss) / max(Loss))
    plt.plot(np.array(Accuracy))
    plt.show()
    IA.save()
    return IA

X, target = sk_dataset.get_dataset()

width = 128
length = 16

# train_data, test_data, train_labels, test_labels = split_data(X, y, 0.9)
IA = NetworkMatrix(nb_input=X.shape[1], width=int(width), nb_output=target.shape[1], length=length, learning_rate=0.08)

IA = create_model(X, target.reshape(-1, 1), IA, nb_iter=int(150))
# IA.load()

# test_data, test_labels = test_data.T, test_labels.T
# result = IA.forward_propagation(test_data)
# print("Shape of result[0]:", result[0].shape)
# print("Type of result[0]:", type(result[0]))
# print(result[len(result) - 1], test_labels)
# pred = IA.get_accuracy(result[len(result) - 1], test_labels)
# print(pred)

