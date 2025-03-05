import numpy as np

from game import Game

class Network:
    def __init__(self, nb_input, width, length, nb_output, learning_rate=0.01):
        self.W = []
        self.b = []
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.9  # Exploration rate
        self.memory = []  # Experience replay buffer
        self.batch_size = 32
        self.max_memory = 5000
        self.max_move_ahead = 10


        self.W.append(np.random.randn(width, nb_input))
        self.b.append(np.random.randn(width, 1))
        
        for i in range(length - 2):
            self.W.append(np.random.randn(width, width))
            self.b.append(np.random.randn(width, 1))
        
        self.W.append(np.random.randn(nb_output, width))
        self.b.append(np.random.randn(nb_output, 1))

    def process_move(self, board):
        best_move = self.get_best_move(board)
        self.train(board, best_move)
        return best_move
    
    def train(self, board, best_move):
        game = Game(board)
        reward = 0
        i = 0
        while not game.finished and i < self.max_move_ahead:
            game.move(best_move)
            reward += self.evaluate(board) * self.ε ** i
            best_move = self.get_best_move(self, board)
            i += 1
        self.back_propagation(reward)
        self.minimization()
    
    def evaluate(self, board):
        return np.sum(board)

    def get_best_move(self, board):
        X = np.array(board).flatten()
        activations = self.forward_propagation(X)
        return np.argmax(activations[-1].T[-1])
    

    # _____IA CORE_____

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
