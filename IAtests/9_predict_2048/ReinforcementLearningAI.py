import numpy as np
from my2048 import Numpy2048
from qnetwork import QNetwork
import random
from collections import deque

class ReinforcementLearningAI:
    def __init__(self, size, hidden_size=256, learning_rate=0.001):
        self.board_size = size
        # Enhanced state representation with additional features
        self.state_size = size * size + 4  # board + additional features
        self.action_size = 4
        
        # Initialize Q-network with improved architecture
        self.q_network = QNetwork(
            input_size=self.state_size,
            hidden_size=hidden_size,
            output_size=self.action_size,
            learning_rate=learning_rate
        )
        
        # Enhanced RL parameters
        self.gamma = 0.99  # Higher discount for longer-term planning
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # Higher minimum exploration
        self.epsilon_decay = 0.999  # Slower decay
        
        # Improved experience replay with prioritization
        self.memory = deque(maxlen=20000)  # Larger memory
        self.batch_size = 64  # Larger batch size
        self.priority_scale = 0.7  # For prioritized sampling
        
        # Move preferences (for tie-breaking)
        self.move_priority = {'up': 3, 'left': 2, 'down': 1, 'right': 0}
        
        self.load()
    
    def get_state(self, board):
        """Enhanced state representation"""
        # Initialize state vector
        state = np.zeros(self.board_size * self.board_size + 4, dtype=float)
        
        # Board state with one-hot encoding for powers of 2
        flat_board = board.flatten()
        for i, value in enumerate(flat_board):
            if value > 0:
                # Convert to log2 and normalize
                power = int(np.log2(value))
                state[i] = power / 11.0  # Normalize by max possible value (2048 = 2^11)
        
        # Additional features
        # 1. Number of empty cells (normalized)
        state[-4] = np.sum(board == 0) / (self.board_size * self.board_size)
        
        # 2. Monotonicity score (how well the tiles are ordered)
        monotonicity = 0
        for i in range(self.board_size):
            # Check rows
            row = np.log2(np.maximum(board[i], 1))
            monotonicity += np.sum(np.diff(row) >= 0)
            # Check columns
            col = np.log2(np.maximum(board[:, i], 1))
            monotonicity += np.sum(np.diff(col) >= 0)
        state[-3] = monotonicity / (2 * self.board_size * (self.board_size - 1))
        
        # 3. Maximum tile (normalized)
        state[-2] = np.log2(np.maximum(np.max(board), 1)) / 11.0
        
        # 4. Smoothness score (similarity between adjacent tiles)
        smoothness = 0
        padded = np.pad(board, 1, mode='constant')
        for i in range(1, padded.shape[0]-1):
            for j in range(1, padded.shape[1]-1):
                if padded[i,j] != 0:
                    neighbors = [padded[i-1,j], padded[i+1,j], padded[i,j-1], padded[i,j+1]]
                    for n in neighbors:
                        if n != 0:
                            smoothness -= abs(np.log2(padded[i,j]) - np.log2(n))
        state[-1] = 1.0 / (1.0 - smoothness / (2 * self.board_size * self.board_size))
        
        return state.reshape(-1, 1)
    
    def monotonicity_score(self, board):
        """Calculate how well the tiles are ordered"""
        score = 0
        
        # Check rows
        for i in range(self.board_size):
            if np.all(np.diff(np.log2(np.maximum(board[i], 1))) >= 0):
                score += 1
            if np.all(np.diff(np.log2(np.maximum(board[i], 1))) <= 0):
                score += 1
        
        # Check columns
        for j in range(self.board_size):
            if np.all(np.diff(np.log2(np.maximum(board[:, j], 1))) >= 0):
                score += 1
            if np.all(np.diff(np.log2(np.maximum(board[:, j], 1))) <= 0):
                score += 1
        
        return score / (4 * self.board_size)
    
    def smoothness_score(self, board):
        """Calculate how smooth the board is (neighboring tiles with similar values)"""
        smooth_score = 0
        log_board = np.log2(np.maximum(board, 1))
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 0:
                    continue
                    
                if i > 0:  # Check vertical neighbor
                    smooth_score -= abs(log_board[i, j] - log_board[i-1, j])
                if j > 0:  # Check horizontal neighbor
                    smooth_score -= abs(log_board[i, j] - log_board[i, j-1])
        
        return 1.0 / (1.0 - smooth_score / (2 * self.board_size ** 2))
    
    def get_valid_moves(self, board):
        """Get list of valid moves with improved ordering"""
        game = Numpy2048(board.shape[0])
        valid_moves = []
        actions = ['up', 'left', 'down', 'right']
        
        # Try each move and simulate outcome
        move_scores = []
        for action in range(4):
            test_game = Numpy2048(board.shape[0])
            test_game.board = board.copy()
            if test_game.move(actions[action]):
                score = (
                    self.monotonicity_score(test_game.board) * 2.0 +
                    self.smoothness_score(test_game.board) +
                    np.sum(test_game.board == 0) * 0.1
                )
                move_scores.append((action, score))
                valid_moves.append(action)
        
        # Sort moves by score
        if move_scores:
            move_scores.sort(key=lambda x: x[1], reverse=True)
            return [m[0] for m in move_scores]
        
        return valid_moves if valid_moves else [0, 1, 2, 3]
    
    def calculate_reward(self, old_board, new_board, old_score, new_score, is_game_over):
        """Enhanced reward calculation"""
        if np.array_equal(old_board, new_board):
            return -2.0  # Increased penalty for invalid moves
        
        if is_game_over:
            return -10.0  # Terminal state penalty
        
        reward = 0.0
        
        # Score-based reward (log-scaled to handle large numbers better)
        score_diff = new_score - old_score
        reward += np.log2(score_diff + 1) if score_diff > 0 else -0.5
        
        # Merging reward (encourage creation of higher value tiles)
        old_max = np.max(old_board)
        new_max = np.max(new_board)
        if new_max > old_max:
            reward += 2.0 * np.log2(new_max)
        
        # Empty cells reward (maintaining open spaces)
        empty_diff = np.sum(new_board == 0) - np.sum(old_board == 0)
        reward += empty_diff * 0.5
        
        # Corner preference (encourage keeping high values in corners)
        corners_old = [old_board[0,0], old_board[0,-1], old_board[-1,0], old_board[-1,-1]]
        corners_new = [new_board[0,0], new_board[0,-1], new_board[-1,0], new_board[-1,-1]]
        if max(corners_new) > max(corners_old):
            reward += np.log2(max(corners_new))
            
        # Monotonicity reward (encourage ordered patterns)
        def get_monotonicity(board):
            mono_score = 0
            for i in range(board.shape[0]):
                row = np.log2(np.maximum(board[i], 1))
                mono_score += np.sum(np.diff(row) >= 0)
                col = np.log2(np.maximum(board[:, i], 1))
                mono_score += np.sum(np.diff(col) >= 0)
            return mono_score
        
        mono_diff = get_monotonicity(new_board) - get_monotonicity(old_board)
        reward += mono_diff * 0.3
        
        return reward
    
    def get_action(self, state, valid_moves):
        """Enhanced action selection with smart exploration"""
        if random.random() < self.epsilon:
            # Weighted random choice favoring "better" moves
            if len(valid_moves) > 1:
                # Prefer up and left moves during exploration
                weights = []
                for move in valid_moves:
                    if move == 0:  # up
                        weights.append(0.4)
                    elif move == 1:  # left
                        weights.append(0.3)
                    else:  # down or right
                        weights.append(0.15)
                weights = [w/sum(weights) for w in weights]
                return np.random.choice(valid_moves, p=weights)
            return random.choice(valid_moves)
        
        q_values = self.q_network.predict(state)
        q_values = q_values.flatten()
        
        # Mask invalid moves
        mask = np.ones_like(q_values) * float('-inf')
        mask[valid_moves] = q_values[valid_moves]
        
        return np.argmax(mask)
    
    def train(self, old_board, action, new_board, old_score, new_score, done):
        """Train using prioritized experience replay and n-step returns"""
        state = self.get_state(old_board)
        next_state = self.get_state(new_board)
        reward = self.calculate_reward(old_board, new_board, old_score, new_score, done)
        
        # Store experience with priority
        priority = abs(reward) + 1e-6  # Priority based on reward magnitude
        self.memory.append((state, action, reward, next_state, done, priority))
        
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch with prioritization
        priorities = np.array([exp[5] for exp in self.memory])
        probs = priorities ** self.priority_scale
        probs /= np.sum(probs)
        
        indices = np.random.choice(
            len(self.memory),
            size=self.batch_size,
            p=probs
        )
        
        batch = [self.memory[i] for i in indices]
        
        states = np.hstack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.hstack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        
        # Double Q-learning update
        current_q = self.q_network.predict(states)
        next_q = self.q_network.predict(next_states)
        
        # Target Q-values with double Q-learning
        targets = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[actions[i], i] = rewards[i]
            else:
                next_action = np.argmax(next_q[:, i])
                targets[actions[i], i] = rewards[i] + self.gamma * next_q[next_action, i]
        
        # Train network
        loss = self.q_network.train(states, targets)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss
    
    def best_move(self, board):
        """Get best move for given board state"""
        state = self.get_state(board)
        valid_moves = self.get_valid_moves(board)
        return self.get_action(state, valid_moves)
    
    def save(self):
        self.q_network.save()
    
    def load(self):
        self.q_network.load()