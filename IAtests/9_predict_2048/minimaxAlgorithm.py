import numpy as np
import random
from copy import deepcopy
from my2048 import Numpy2048

def evaluate_board(board):
    # Evaluate board state - higher values are better
    score = 0
    # Reward high values
    score += np.sum(board) * 0.1
    # Reward empty cells
    score += np.sum(board == 0) * 100
    # Penalize scattered high values
    diff_horizontal = np.abs(np.diff(board, axis=1))
    diff_vertical = np.abs(np.diff(board, axis=0))
    score -= (np.sum(diff_horizontal) + np.sum(diff_vertical)) * 0.05
    return score

def minimax(game, depth, is_maximizing, moves):
    if depth == 0 or game.is_game_over():
        return evaluate_board(game.board), None

    if is_maximizing:
        max_eval = float('-inf')
        best_move = None
        
        for move_idx, move_dir in moves.items():
            game_copy = Numpy2048(game.board.shape[0])
            game_copy.board = game.board.copy()
            game_copy.score = game.score
            
            if game_copy.move(move_dir):
                eval_score, _ = minimax(game_copy, depth - 1, False, moves)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move_idx
        
        return max_eval, best_move
    else:
        # Simulate possible new tile placements (2 or 4)
        min_eval = float('inf')
        empty_cells = np.where(game.board == 0)
        empty_positions = list(zip(empty_cells[0], empty_cells[1]))
        
        if not empty_positions:
            return evaluate_board(game.board), None
            
        # Sample a subset of empty positions to reduce branching factor
        num_samples = min(3, len(empty_positions))
        sampled_positions = random.sample(empty_positions, num_samples)
        
        for pos in sampled_positions:
            for new_tile in [2, 4]:
                game_copy = Numpy2048(game.board.shape[0])
                game_copy.board = game.board.copy()
                game_copy.score = game.score
                game_copy.board[pos] = new_tile
                
                eval_score, _ = minimax(game_copy, depth - 1, True, moves)
                min_eval = min(min_eval, eval_score)
        
        return min_eval, None

def best_move(board, depth=1):
    game = Numpy2048(board.shape[0])
    game.board = board.copy()
    moves = {0: 'up', 1: 'left', 2: 'down', 3: 'right'}
    
    # Start the minimax search
    _, best_move_idx = minimax(game, depth, True, moves)
    return best_move_idx if best_move_idx is not None else random.randint(0, 3)