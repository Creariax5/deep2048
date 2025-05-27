import numpy as np
import matplotlib.pyplot as plt

def create_model(X, y, IA, nb_iter = 100):
    X, y = X.T, y.T
    Loss, Accuracy = IA.train(X, y, nb_iter)
    
    # plt.plot(np.array(Loss) / max(Loss))
    # plt.plot(np.array(Accuracy))
    # plt.show()
    IA.save()
    return IA
