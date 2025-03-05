import numpy as np

class Game:
    def __init__(self, board):
        self.board = board
        self.finished = False
        self.score = 0
        self.size = board.shape[0]

    def move(self, direction):
        new_board, score_gain = self.calculate_move(direction)

        if np.array_equal(self.board, new_board):
            self.finished = True
            return False
        
        self.board = new_board
        self.score += score_gain
        self.add_new_tile(1)
        return True

    def add_new_tile(self, count=1):
        empty_positions = np.argwhere(self.board == 0)
        if empty_positions.size > 0:
            chosen_positions = empty_positions[np.random.choice(len(empty_positions), 
                                                size=min(count, len(empty_positions)), 
                                                replace=False)]
            for x, y in chosen_positions:
                self.board[x, y] = np.random.choice([2, 4], p=[0.9, 0.1])

    def compress(self, line):
        """Merge a row or column by shifting nonzero values left and combining equal adjacent numbers"""
        non_zero = line[line != 0]  # Remove zeros
        merged = []
        score_gained = 0
        i = 0

        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged.append(non_zero[i] * 2)
                score_gained += non_zero[i] * 2
                i += 2  # Skip next element as it’s merged
            else:
                merged.append(non_zero[i])
                i += 1

        return np.array(merged + [0] * (self.size - len(merged)), dtype=np.int32), score_gained

    def calculate_move(self, direction):
        """Computes the new board state and score after a move"""
        transformations = {
            'left': lambda b: b,
            'right': lambda b: np.fliplr(b),
            'up': lambda b: b.T,
            'down': lambda b: np.flipud(b.T)
        }

        transformed_board = transformations[direction](self.board)
        new_board = np.zeros_like(transformed_board)
        score_gain = 0

        for i in range(self.size):
            new_board[i], gained = self.compress(transformed_board[i])
            score_gain += gained

        # Reverse the transformation
        reverse_transformations = {
            'left': lambda b: b,
            'right': lambda b: np.fliplr(b),
            'up': lambda b: b.T,
            'down': lambda b: np.flipud(b.T)
        }

        return reverse_transformations[direction](new_board), score_gain
    
    def display(self):
        print('\n'.join([' '.join(f'{cell:4}' for cell in row) for row in self.board]))
        print(f'Score: {self.score}')
        if self.finished:
            print("Game Over - No valid moves left!")
