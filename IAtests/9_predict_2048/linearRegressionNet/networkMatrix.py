import numpy as np
import shelve
from tqdm import tqdm

class NetworkMatrix:
    def __init__(self, nb_input, width, nb_output, length, learning_rate=0.01):
        self.W = []
        self.b = []

        # Xavier/Glorot initialization with smaller variance
        self.W.append(np.random.randn(width, nb_input) * np.sqrt(1.0 / nb_input))
        self.b.append(np.zeros((width, 1)))
        
        for i in range(length - 2):
            self.W.append(np.random.randn(width, width) * np.sqrt(1.0 / width))
            self.b.append(np.zeros((width, 1)))
        
        self.W.append(np.random.randn(nb_output, width) * np.sqrt(1.0 / width))
        self.b.append(np.zeros((nb_output, 1)))
        
        self.learning_rate = learning_rate
        self.epsilon = 1e-7  # Small constant to prevent division by zero

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        # Clip values to prevent exploding gradients
        Z = np.clip(Z, -1e6, 1e6)
        return (Z > 0).astype(float)
    
    def forward_propagation(self, X):
        activations = [X]
        Z_values = []

        for i in range(len(self.W) - 1):
            Z = self.W[i].dot(activations[i]) + self.b[i]
            Z = np.clip(Z, -1e6, 1e6)  # Keep this for numerical stability
            Z_values.append(Z)
            activations.append(self.relu(Z))

        # Output layer (linear activation for regression)
        Z = self.W[-1].dot(activations[-1]) + self.b[-1]
        Z = np.clip(Z, -1e6, 1e6)  # Keep this for numerical stability
        Z_values.append(Z)
        activations.append(Z)
        return activations, Z_values

    def error(self, A, y):
        # Clip prediction values to prevent extreme errors
        A_clipped = np.clip(A, -1e6, 1e6)
        return (y - A_clipped)**2
    
    def back_propagation(self, activations, Z_values, X, y):
        gradients = []
        m = y.shape[1]

        dZ = -2/m * (y - activations[-1])

        for i in range(len(activations)-2, -1, -1):
            dW = 1 / m * dZ.dot(activations[i].T)
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            
            gradients.append((dW, db))
            
            if i >= 1:
                relu_deriv = self.relu_derivative(Z_values[i-1])
                dZ = np.dot(self.W[i].T, dZ) * relu_deriv
                dZ = np.clip(dZ, -1e6, 1e6)  # Keep this for stability

        gradients.reverse()
        return gradients

    def minimization(self, gradients):
        for i in range(len(gradients)):
            # Apply gradient clipping
            dW = gradients[i][0] * self.learning_rate
            db = gradients[i][1] * self.learning_rate
            
            self.W[i] = self.W[i] - dW
            self.b[i] = self.b[i] - db
        return (self.W, self.b)
    
    def calculate_loss(self, predictions, y):
        # print("loss: ", (predictions - y) ** 2)
        mse = np.mean((predictions - y) ** 2)
        return mse
    
    def train(self, X, y):
        activations, Z_values = self.forward_propagation(X)
        gradients = self.back_propagation(activations, Z_values, X, y)
        self.minimization(gradients)

        # print("activations: ", activations[-1])
        # print("expected: ", y)

        Loss = self.calculate_loss(activations[-1], y)
        return Loss
    
    def get_accuracy(self, A, y):
        # Clip predictions to prevent invalid values
        A_clipped = np.clip(A, -1e6, 1e6)
        predictions = (A_clipped > 0.5).astype(int)
        return np.mean(predictions == y) * 100

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