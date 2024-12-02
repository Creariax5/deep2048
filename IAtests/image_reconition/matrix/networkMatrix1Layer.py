import numpy as np
import shelve

class NetworkMatrix:
    def __init__(self, nb_input, width, nb_output, length, learning_rate=0.01):
        self.W = []
        self.b = []

        self.W.append(np.random.randn(width, nb_input))
        self.b.append(np.random.randn(width, 1))
        # print(self.W[0].shape)
        # self.W_hidden = np.random.randn(width, width)
        # self.b_hidden = np.random.randn(width)
        # self.W_out = np.random.randn(width, nb_output)
        self.W.append(np.random.randn(nb_output, width))
        self.b.append(np.random.randn(nb_output, 1))
        self.learning_rate = learning_rate
    
    def forward_propagation(self, X):
        activations = []

        Z = self.W[0].dot(X) + self.b[0]
        activations.append(self.sigmoid(Z))
        Z = self.W[1].dot(activations[0]) + self.b[1]
        activations.append(self.sigmoid(Z))
        return activations

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def log_loss(self, A, y):
        m = len(y)
        return -1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
    
    def back_propagation(self, activations, X, y):
        gradients = []

        m = y.shape[1]

        dZ2 = activations[1] - y
        dW = 1 / m * dZ2.dot(activations[0].T)
        db = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        gradients.append((dW, db))
        
        dZ1 = np.dot(self.W[1].T, dZ2) * activations[0] * (1 - activations[0])
        dW = 1 / m * dZ1.dot(X.T)
        db = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
        gradients.append((dW, db))
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

        for i in range(nb_iter):
            activations = self.forward_propagation(X)
            Loss.append(self.log_loss(activations[1], y))
            a = self.get_accuracy(activations[1], y)
            print(a, " %")
            Accuracy.append(a)
            gradients = self.back_propagation(activations, X, y)
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
        with shelve.open('network') as db:
            self.W = db['W']
            self.b = db['b']
