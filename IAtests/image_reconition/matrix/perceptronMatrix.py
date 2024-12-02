import numpy as np
import shelve

class PerceptronMatrix:
    def __init__(self, nb_input, learning_rate=0.01):
        self.W = np.random.randn(nb_input, 1) * 0.001
        self.b = np.random.randn(1)
        self.learning_rate = learning_rate
    
    def forward_propagation(self, X):
        Z = X.dot(self.W) + self.b
        A = self.activation(Z)
        return A

    def activation(self, Z):
        return 1 / (1 + np.exp(-Z))

    def log_loss(self, A, y):
        m = len(y)
        return -1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
    
    def gradients(self, A, X, y):
        m = len(y)
        dW = 1 / m * np.dot(X.T, A - y)
        db = 1 / m * np.sum(A - y)
        return (dW, db)

    def minimization(self, dW, db):
        self.W = self.W - dW * self.learning_rate
        self.b = self.b - db * self.learning_rate
        return (self.W, self.b)
    
    def train(self, X, y, nb_iter):
        Loss = []
        Accuracy = []

        for i in range(nb_iter):
            A = self.forward_propagation(X)
            Loss.append(self.log_loss(A, y))
            Accuracy.append(self.get_accuracy(A, y))
            dW, db = self.gradients(A, X, y)
            self.minimization(dW, db)
        return Loss, Accuracy
    
    def get_accuracy(self, A, y):
        predictions = (A > 0.5).astype(int)  # Convert probabilities to binary predictions
        return np.mean(predictions == y)     # Calculate proportion of correct predictions

    def save(self):
        with shelve.open('mydata') as db:
            db['W'] = self.W
            db['b'] = self.b
    
    def load(self):
        with shelve.open('mydata') as db:
            self.W = db['W']
            self.b = db['b']
