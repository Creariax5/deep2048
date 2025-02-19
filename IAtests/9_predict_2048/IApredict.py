import time
import numpy as np
from linearRegressionNet.networkMatrix import NetworkMatrix

import numpy as np
import random
from my2048 import Numpy2048

class IApredict:
    def __init__(self, size, width=32, length=8, learning_rate=0.01):
        self.IA = NetworkMatrix(nb_input=size*size, width=width, nb_output=1, length=length, learning_rate=learning_rate)
        self.IA.load()

    def predict_final_score(self, X):
        X = X.reshape(X.shape[0]**2, 1)
        output = self.IA.forward_propagation(X)[-1][-1]
        return output[-1]

    def train(self, X, y):
        X = np.array(X)
        X_reshaped = np.array([board.flatten() for board in X]).T
        y_reshaped = np.full((1, len(X)), y)
        # print("X_reshaped", X_reshaped, "y_reshaped", y_reshaped)
        Loss = self.IA.train(X_reshaped, y_reshaped)
        return Loss

    def simple_minimax(self, game, moves):
        max_eval = float('-inf')
        best_move = None
        
        # Try each possible move
        for move_idx, move_dir in moves.items():
            # Create a copy of the game state
            game_copy = type(game)(game.board.shape[0])
            game_copy.board = game.board.copy()
            
            # If move is valid, evaluate resulting position
            if game_copy.move(move_dir):
                score = self.predict_final_score(game_copy.board)
                if score > max_eval:
                    max_eval = score
                    best_move = move_idx
        
        return best_move

    def best_move(self, board):
        game = Numpy2048(board.shape[0])
        game.board = board.copy()
        moves = {0: 'up', 1: 'left', 2: 'down', 3: 'right'}
        
        # Start the minimax search
        best_move_idx = self.simple_minimax(game, moves)
        return best_move_idx if best_move_idx is not None else random.randint(0, 3)
