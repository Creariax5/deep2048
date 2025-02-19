import numpy as np
import shelve
from tqdm import tqdm

class NetworkMatrix:
    def __init__(self, nb_input, width, nb_output, length, learning_rate=0.01):
        self.W = []
        self.b = []

        # Xavier/Glorot initialization for better convergence
        self.W.append(np.random.randn(width, nb_input) * np.sqrt(2.0 / nb_input))
        self.b.append(np.zeros((width, 1)))  # Initialize biases to zero
        
        for i in range(length - 2):
            self.W.append(np.random.randn(width, width) * np.sqrt(2.0 / width))
            self.b.append(np.zeros((width, 1)))
        
        self.W.append(np.random.randn(nb_output, width) * np.sqrt(2.0 / width))
        self.b.append(np.zeros((nb_output, 1)))
        
        self.learning_rate = learning_rate

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return Z > 0
    
    def forward_propagation(self, X):
        activations = [X]
        Z_values = []

        for i in range(len(self.W) - 1):
            Z = self.W[i].dot(activations[i]) + self.b[i]
            Z_values.append(Z)
            activations.append(self.relu(Z))

        # Output layer (linear activation for regression)
        Z = self.W[-1].dot(activations[-1]) + self.b[-1]
        Z_values.append(Z)
        activations.append(Z)  # Linear activation for regression
        return activations, Z_values

    def error(self, A, y):
        return (y - A)**2
    
    def back_propagation(self, activations, Z_values, X, y):
        gradients = []

        m = y.shape[1]

        dZ = -2/m * (y - activations[-1])

        for i in range(len(activations)-2 , -1, -1):
            dW = 1 / m * dZ.dot(activations[i].T)
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            gradients.append((dW, db))
            if i >= 1:
                dZ = np.dot(self.W[i].T, dZ) * self.relu_derivative(Z_values[i-1])

        gradients.reverse()
        return gradients

    def minimization(self, gradients):
        for i in range(len(gradients)):
            self.W[i] = self.W[i] - gradients[i][0] * self.learning_rate
            self.b[i] = self.b[i] - gradients[i][1] * self.learning_rate
        return (self.W, self.b)
    
    def train(self, X, y, nb_iter=1):
        Loss = []
        Accuracy = []

        for i in range(nb_iter):
            activations, Z_values = self.forward_propagation(X)
            # Loss.append(self.log_loss(activations[-1], y))
            # if i % 200 == 0:
            #     a = self.get_accuracy(activations[-1], y)
            #     print(a, "%")
            #     show("newData.png", self, X, y)
            # Accuracy.append(a)
            gradients = self.back_propagation(activations, Z_values, X, y)
            self.minimization(gradients)
        return Loss, Accuracy
    
    def get_accuracy(self, A, y):
        predictions = (A > 0.5).astype(int)
        return np.mean(predictions == y)

    def save(self):
        with shelve.open('network') as db:
            db['W'] = self.W
            db['b'] = self.b
    
    def load(self):
        try:
            with shelve.open('network') as db:
                self.W = db['W']
                self.b = db['b']
            print("NetworkMatrix.load: network loaded")
        except Exception as e:
            print("NetworkMatrix.load: no network found")
