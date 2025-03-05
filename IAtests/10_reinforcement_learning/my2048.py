from copy import deepcopy
import random
import numpy as np

class Numpy2048:
    def __init__(self, size=4):
        self.size = size
        self.score = 0
        self.board = np.zeros((size, size), dtype=np.int32)
        self.add_new_tile(2)
        self.game_over = False
    
    def add_new_tile(self, count=1):
        empty_cells = np.where(self.board == 0)
        empty_positions = list(zip(empty_cells[0], empty_cells[1]))
        
        if empty_positions:
            positions = np.random.choice(len(empty_positions), 
                                      size=min(count, len(empty_positions)), 
                                      replace=False)
            
            for pos in positions:
                x, y = empty_positions[pos]
                self.board[x, y] = np.random.choice([2, 4], p=[0.9, 0.1])
    
    def compress(self, line):
        line = line[line != 0]
        if len(line) == 0:
            return np.zeros(self.size, dtype=np.int32), 0
            
        result = np.zeros(self.size, dtype=np.int32)
        write_pos = 0
        i = 0
        score_gained = 0
        
        while i < len(line):
            if i + 1 < len(line) and line[i] == line[i + 1]:
                result[write_pos] = line[i] * 2
                score_gained += line[i] * 2
                i += 2
            else:
                result[write_pos] = line[i]
                i += 1
            write_pos += 1
            
        return result, score_gained
    
    def calculate_move(self, direction):
        new_board = self.board.copy()
        score_gain = 0
        
        if direction == 'left':
            board = new_board
        elif direction == 'right':
            board = np.flip(new_board, axis=1)
        elif direction == 'up':
            board = new_board.T
        else:  # down
            board = np.flip(new_board.T, axis=1)
        
        for i in range(self.size):
            board[i], gained = self.compress(board[i])
            score_gain += gained
        
        if direction == 'right':
            new_board = np.flip(board, axis=1)
        elif direction == 'up':
            new_board = board.T
        elif direction == 'down':
            new_board = np.flip(board, axis=1).T
        else:
            new_board = board
            
        return new_board, score_gain
    
    def move(self, direction):
        new_board, score_gain = self.calculate_move(direction)
        
        if np.array_equal(self.board, new_board):
            self.game_over = True
            return False
        
        self.board = new_board
        self.score += score_gain
        self.add_new_tile(1)
        return True
    
    def is_game_over(self):
        return self.game_over or (
            np.all(self.board != 0) and
            not any(self.board[i, j] == self.board[i, j + 1]
                   for i in range(self.size)
                   for j in range(self.size - 1)) and
            not any(self.board[i, j] == self.board[i + 1, j]
                   for i in range(self.size - 1)
                   for j in range(self.size))
        )
    
    def display(self):
        print('\n'.join([' '.join(f'{cell:4}' for cell in row) for row in self.board]))
        print(f'Score: {self.score}')
        if self.game_over:
            print("Game Over - No valid moves left!")

def random_move(board):
    return random.randint(0, 3)

def play_game(size=4, mode=random_move):
    import time
    start_time = time.time()
    moves_count = 0

    all_game = []
    game = Numpy2048(size)
    moves = {0: 'up', 1: 'left', 2: 'down', 3: 'right'}

    while not game.is_game_over():
        all_game.append(deepcopy(game.board))
        move = mode(game.board)
        if move in moves:
            if game.move(moves[move]):
                moves_count += 1
    # game.display()
    
    end_time = time.time()
    duration = end_time - start_time
    
    return {
        'duration': duration,
        'moves': moves_count,
        'score': game.score,
        'moves_per_second': moves_count/duration,
        'boards': all_game
    }