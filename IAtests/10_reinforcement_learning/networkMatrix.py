import numpy as np
import shelve
from tqdm import tqdm

class NetworkMatrix:
    def __init__(self, nb_input, width, nb_output, length, learning_rate=0.01):
        self.W = []
        self.b = []

        self.W.append(np.random.randn(width, nb_input))
        self.b.append(np.random.randn(width, 1))
        
        for i in range(length - 2):
            self.W.append(np.random.randn(width, width))
            self.b.append(np.random.randn(width, 1))
        
        self.W.append(np.random.randn(nb_output, width))
        self.b.append(np.random.randn(nb_output, 1))
        
        self.learning_rate = learning_rate

    def forward_propagation(self, X):
        activations = [X]

        for i in range(len(self.W) - 1):
            Z = self.W[i].dot(activations[i]) + self.b[i]
            activations.append(self.sigmoid(Z))
        
        # linear activation for output
        Z = self.W[-1].dot(activations[-1]) + self.b[-1]
        activations.append(Z)
        return activations

    def sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))

    def log_loss(self, A, y):
        m = len(y)
        epsilon = 1e-15
        A = np.clip(A, epsilon, 1 - epsilon)
        return -1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
    
    def back_propagation(self, activations, y):
        gradients = []

        m = y.shape[1]

        dZ = activations[-1] - y

        for i in range(len(activations)-2 , -1, -1):
            dW = 1 / m * dZ.dot(activations[i].T)
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            gradients.append((dW, db))
            if i > 0:
                dZ = np.dot(self.W[i].T, dZ) * activations[i] * (1 - activations[i])

        gradients.reverse()
        return gradients

    def minimization(self, gradients):
        for i in range(len(gradients)):
            self.W[i] = self.W[i] - gradients[i][0] * self.learning_rate
            self.b[i] = self.b[i] - gradients[i][1] * self.learning_rate
        return (self.W, self.b)
    
    def train(self, X, y, nb_iter):
        Loss = []
        Accuracy = []

        for i in tqdm(range(nb_iter)):
            activations = self.forward_propagation(X)
            Loss.append(self.log_loss(activations[-1], y))
            a = self.get_accuracy(activations[-1], y)
            if i % 200 == 0:
                print(a, "%")
                # show("newData.png", self, X, y)
            Accuracy.append(a)
            gradients = self.back_propagation(activations, y)
            self.minimization(gradients)
        return Loss, Accuracy
    
    def get_accuracy(self, A, y):
        # Get the predicted class (maximum probability)
        predictions = np.argmax(A, axis=0)
        true_labels = np.argmax(y, axis=0)
        
        # Compare predictions with true labels
        return np.mean(predictions == true_labels)

    def save(self):
        with shelve.open('network') as db:
            db['W'] = self.W
            db['b'] = self.b
    
    def load(self):
        with shelve.open('network') as db:
            self.W = db['W']
            self.b = db['b']
