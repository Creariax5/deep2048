from matplotlib import pyplot as plt
import numpy as np
import sk_dataset
from networkMatrix import NetworkMatrix
import my_dataset

def create_model(X, y, IA: NetworkMatrix, nb_iter = 100):
    X, y = X.T, y.T
    Loss, Accuracy = IA.train(X, y, nb_iter)
    
    plt.plot(np.array(Loss) / max(Loss))
    plt.plot(np.array(Accuracy))
    plt.show()
    IA.save()
    return IA

def test_IA(X, y, IA):
    X, y = X.T, y.T
    activations = IA.forward_propagation(X)
    a = IA.get_accuracy(activations[-1], y)
    prediction = sk_dataset.convert_from_one_hot(activations[-1].T)[:10]
    target = sk_dataset.convert_from_one_hot(y.T)[:10]
    print("prediction   : ", prediction)
    print("target       : ", target)
    print("score        : ", a, "%")

width = 128
length = 3

X, y = my_dataset.get_digits()

train_data, test_data, train_labels, test_labels = sk_dataset.split_data(X, y, 0.9)

IA = NetworkMatrix(nb_input=train_data.shape[1], width=int(width), nb_output=train_labels.shape[1], length=length, learning_rate=0.1)

IA = create_model(train_data, train_labels, IA, nb_iter=int(4000))
# IA.load()
test_IA(test_data, test_labels, IA)
