import numpy as np
from linearRegressionNet.networkMatrix import NetworkMatrix
from my2048 import Numpy2048
import random

class ReinforcementLearningAI:
    def __init__(self, size, width=32, length=4, learning_rate=0.01, gamma=0.95, epsilon=0.1):
        self.board_size = size
        self.state_size = size * size
        self.action_size = 4  # up, left, down, right
        
        # Q-network: state -> action values
        self.q_network = NetworkMatrix(
            nb_input=self.state_size,
            width=width,
            nb_output=self.action_size,
            length=length,
            learning_rate=learning_rate
        )
        
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.memory = []  # replay memory
        self.memory_size = 10000
        self.batch_size = 32
        
    def get_state(self, board):
        """Convert board to network input format"""
        return board.flatten().reshape(self.state_size, 1)
    
    def get_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, 3)
        
        # Get Q-values for all actions
        q_values, _ = self.q_network.forward_propagation(state)
        return np.argmax(q_values[-1])
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=None):
        """Train Q-network on random batch from replay memory"""
        if batch_size is None:
            batch_size = min(self.batch_size, len(self.memory))
        
        if len(self.memory) < batch_size:
            return 0
        
        batch = random.sample(self.memory, batch_size)
        total_loss = 0
        
        for state, action, reward, next_state, done in batch:
            # Get current Q-values
            current_q, _ = self.q_network.forward_propagation(state)
            current_q = current_q[-1]
            
            # Get target Q-values
            if done:
                target_q = reward
            else:
                next_q, _ = self.q_network.forward_propagation(next_state)
                target_q = reward + self.gamma * np.max(next_q[-1])
            
            # Update Q-value for taken action
            target = current_q.copy()
            target[action] = target_q
            
            # Train network
            loss = self.q_network.train(state, target.reshape(-1, 1))
            total_loss += loss
        
        return total_loss / batch_size
    
    def best_move(self, board):
        """Get best move for given board state"""
        state = self.get_state(board)
        return self.get_action(state, training=False)
    
    def train(self, board, action, reward, next_board, done):
        """Train the AI on a single step"""
        state = self.get_state(board)
        next_state = self.get_state(next_board)
        self.remember(state, action, reward, next_state, done)
        return self.replay()
    
    def save(self):
        """Save Q-network weights"""
        self.q_network.save()
    
    def load(self):
        """Load Q-network weights"""
        self.q_network.load()
