import numpy as np
from scipy.ndimage import uniform_filter1d

def moyenne_mobile_scipy(scores, taille_fenetre=10):
    return uniform_filter1d(scores, size=taille_fenetre, mode='nearest')

def create_board(size):
    return np.zeros((size, size))

def evaluate(board):
        return np.sum(board)

def enhanced_evaluate(board, prev_board=None):
    """
    Enhanced reward function that considers:
    1. Merges (higher reward for merging larger tiles)
    2. Empty spaces (reward for maintaining open spaces)
    3. Monotonicity (reward for keeping tiles in order)
    """
    # Base reward for empty spaces
    empty_spaces = np.count_nonzero(board == 0)
    empty_reward = empty_spaces / 16.0  # Normalize by total cells
    
    # Reward for having large values (using log scale)
    value_reward = np.sum(np.log2(np.maximum(board, 2))) / 64.0  # Normalize
    
    # If we have previous board, calculate merge rewards
    merge_reward = 0
    if prev_board is not None:
        # Detect merges by comparing sum of values
        prev_sum = np.sum(prev_board)
        current_sum = np.sum(board)
        if current_sum > prev_sum:
            # Log-scale reward for merges (larger merges = higher rewards)
            merge_reward = np.log2(current_sum - prev_sum) / 10.0
    
    # Monotonicity reward (tiles should increase/decrease in order)
    monotonicity_reward = 0
    # Check rows
    for i in range(4):
        row = board[i, :]
        if (np.diff(row) >= 0).all() or (np.diff(row) <= 0).all():
            monotonicity_reward += 0.1
    # Check columns
    for i in range(4):
        col = board[:, i]
        if (np.diff(col) >= 0).all() or (np.diff(col) <= 0).all():
            monotonicity_reward += 0.1
            
    # Combine rewards with appropriate weights
    total_reward = (
        0.3 * empty_reward +   # Reward for empty spaces
        0.3 * value_reward +   # Reward for high values
        0.3 * merge_reward +   # Reward for merges
        0.1 * monotonicity_reward  # Reward for organized board
    )
    
    return total_reward
