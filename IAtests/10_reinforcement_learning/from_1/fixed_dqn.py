import numpy as np
import random
from collections import deque

class DQNAgent:
    """Simplified DQN Agent with robust learning for 2048"""
    def __init__(self, state_size, action_size, learning_rate=0.0001):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters - optimized for 2048
        self.gamma = 0.95          # Discount factor
        self.epsilon = 1.0         # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        
        # Replay memory
        self.memory = deque(maxlen=20000)
        self.batch_size = 32
        
        # Build network - intentionally simple
        self.build_model()
        
        # Metrics for monitoring learning
        self.loss_history = []
        self.q_history = []
        self.reward_history = []
        
    def build_model(self):
        """Build a simple but effective neural network"""
        # Initialize weights with Xavier/Glorot initialization
        # Single hidden layer (simple but effective for 2048)
        hidden_size = 64
        
        # Input layer to hidden layer
        self.W1 = np.random.randn(hidden_size, self.state_size) * np.sqrt(2.0 / self.state_size)
        self.b1 = np.zeros((hidden_size, 1))
        
        # Hidden layer to output layer
        self.W2 = np.random.randn(self.action_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((self.action_size, 1))
        
        # Target network (exact copy)
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
        
        # Training counter for target network updates
        self.train_counter = 0
        self.target_update_freq = 10
        
    def preprocess_state(self, state):
        """Preprocess state for the network"""
        # Convert to array and log-transform
        state_array = np.array(state, dtype=np.float32).flatten().reshape(-1, 1)
        
        # Log2 transform (handle zeros properly)
        processed = np.zeros_like(state_array)
        non_zero = state_array > 0
        processed[non_zero] = np.log2(state_array[non_zero])
        
        # Normalize
        max_val = 16.0  # log2 of maximum reasonable tile (65536)
        processed = processed / max_val
        
        return processed
        
    def forward(self, state, target=False):
        """Forward pass through the network"""
        # Use target network if specified
        if target:
            W1, b1 = self.target_W1, self.target_b1
            W2, b2 = self.target_W2, self.target_b2
        else:
            W1, b1 = self.W1, self.b1
            W2, b2 = self.W2, self.b2
            
        # First layer
        z1 = np.dot(W1, state) + b1
        a1 = np.maximum(0, z1)  # ReLU
        
        # Output layer
        z2 = np.dot(W2, a1) + b2
        
        return z2, a1
        
    def update_target_network(self):
        """Update target network weights"""
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
        # print("Target network updated")
        
    def remember(self, state, action, reward, next_state, done):
        """Remember experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        self.reward_history.append(reward)
        
    def act(self, state, legal_moves=None):
        """Choose action using epsilon-greedy policy"""
        if legal_moves is None:
            legal_moves = [0, 1, 2, 3]
            
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_moves)
            
        # Get action values
        q_values, _ = self.forward(self.preprocess_state(state))
        q_values = q_values.flatten()
        
        # Filter by legal moves
        legal_q_values = [(move, q_values[move]) for move in legal_moves]
        
        # Choose best legal move
        return max(legal_q_values, key=lambda x: x[1])[0]
        
    def replay(self, batch_size=None):
        """Train the network using experience replay"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return 0
            
        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        total_loss = 0
        
        for state, action, reward, next_state, done in minibatch:
            # Preprocess states
            state_vector = self.preprocess_state(state)
            next_state_vector = self.preprocess_state(next_state)
            
            # Get current Q values
            current_q, a1 = self.forward(state_vector)
            
            # Get target Q values
            target_q = current_q.copy()
            
            if done:
                target_q[action] = reward
            else:
                # Double DQN: select action using main network
                next_q, _ = self.forward(next_state_vector)
                best_action = np.argmax(next_q)
                
                # Evaluate action using target network
                next_target_q, _ = self.forward(next_state_vector, target=True)
                target_q[action] = reward + self.gamma * next_target_q[best_action]
            
            # Calculate TD error
            td_error = current_q[action] - target_q[action]
            total_loss += td_error ** 2
            
            # Calculate gradients (manually)
            dq = current_q.copy()
            dq[action] = td_error  # Only propagate error for the taken action
            
            # Backpropagation (second layer)
            dW2 = np.dot(dq, a1.T)
            db2 = dq
            
            # Backpropagation (first layer)
            dz1 = np.dot(self.W2.T, dq)
            da1 = dz1 * (a1 > 0)  # ReLU derivative
            dW1 = np.dot(da1, state_vector.T)
            db1 = da1
            
            # Gradient clipping to prevent exploding gradients
            clip_value = 1.0
            dW1 = np.clip(dW1, -clip_value, clip_value)
            db1 = np.clip(db1, -clip_value, clip_value)
            dW2 = np.clip(dW2, -clip_value, clip_value)
            db2 = np.clip(db2, -clip_value, clip_value)
            
            # Update weights
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
        
        # Occasionally update target network
        self.train_counter += 1
        if self.train_counter % self.target_update_freq == 0:
            self.update_target_network()
            
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Record average loss for monitoring
        avg_loss = total_loss / batch_size
        # Fix: Convert numpy array to float if necessary
        if isinstance(avg_loss, np.ndarray):
            avg_loss = float(avg_loss)
        self.loss_history.append(avg_loss)
        
        # Log average Q-value (debugging)
        if len(minibatch) > 0:
            avg_q_val = np.mean([self.forward(self.preprocess_state(s))[0][a] for s, a, _, _, _ in minibatch])
            # Fix: Convert numpy array to float if necessary
            if isinstance(avg_q_val, np.ndarray):
                avg_q_val = float(avg_q_val)
            self.q_history.append(avg_q_val)
            
        return avg_loss
        
    def save_model(self, filename):
        """Save model weights to file"""
        np.savez(filename, 
                 W1=self.W1, 
                 b1=self.b1, 
                 W2=self.W2, 
                 b2=self.b2)
        
    def load_model(self, filename):
        """Load model weights from file"""
        data = np.load(filename)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.update_target_network()
        
    def get_learning_stats(self):
        """Get learning statistics for monitoring"""
        return {
            'loss_history': self.loss_history,
            'q_history': self.q_history,
            'reward_history': self.reward_history
        }


def effective_reward_function(board, prev_board=None, score_increase=0):
    """Effective reward function for 2048 with clear learning signal"""
    if prev_board is None:
        return 0
        
    # Part 1: Basic reward based on score increase
    merge_reward = 0
    if score_increase > 0:
        merge_reward = np.log2(score_increase) * 0.1
        
    # Part 2: Empty space reward (crucial for 2048)
    empty_spaces = np.count_nonzero(board == 0)
    empty_spaces_prev = np.count_nonzero(prev_board == 0)
    empty_delta = empty_spaces - empty_spaces_prev
    empty_reward = 0.1 * empty_delta + 0.01 * empty_spaces
    
    # Part 3: Max tile reward (breakthrough detection)
    max_tile = np.max(board)
    max_tile_prev = np.max(prev_board)
    max_tile_reward = 0
    if max_tile > max_tile_prev:
        max_tile_reward = np.log2(max_tile) * 0.5  # Strong reward for breakthroughs
        
    # Part 4: Monotonicity reward (coherent structure)
    monotonicity_reward = 0
    
    # Check rows for monotonicity
    for i in range(board.shape[0]):
        row = board[i, :]
        # Count increasing or decreasing sequences
        if np.all(np.diff(row) >= 0) or np.all(np.diff(row) <= 0):
            monotonicity_reward += 0.1
            
    # Check columns for monotonicity
    for i in range(board.shape[1]):
        col = board[:, i]
        if np.all(np.diff(col) >= 0) or np.all(np.diff(col) <= 0):
            monotonicity_reward += 0.1
    
    # Sum rewards with appropriate weights
    total_reward = merge_reward + empty_reward + max_tile_reward + monotonicity_reward
    
    # Add regularization to avoid too small or too large rewards
    total_reward = np.clip(total_reward, -1.0, 1.0)
    
    return total_reward