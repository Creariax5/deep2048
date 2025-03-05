from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import time

from game import Game
from utils import create_board, moyenne_mobile_scipy
from fixed_dqn import DQNAgent, effective_reward_function

# Game parameters
BOARD_SIZE = 4
ITER = 5000  # Start with a reasonable number of episodes
moves_list = ['up', 'left', 'down', 'right']

# Create agent
state_size = BOARD_SIZE * BOARD_SIZE
action_size = len(moves_list)
agent = DQNAgent(state_size, action_size, learning_rate=0.0001)  # Very small learning rate for stability

# Function to get legal moves
def get_legal_moves(game, prev_board=None):
    """Get legal moves and avoid loops"""
    legal_moves = []
    
    # Try each move
    for move_idx, move in enumerate(moves_list):
        # Create a copy to test the move
        test_game = Game(game.board.copy())
        old_board = test_game.board.copy()
        
        # Try the move
        test_game.move(move)
        
        # If the board changed, it's a legal move
        if not np.array_equal(test_game.board, old_board):
            if prev_board is not None and np.array_equal(test_game.board, prev_board):
                # This would create a loop, avoid if possible
                continue
            legal_moves.append(move_idx)
    
    # If no legal moves found (shouldn't happen in a valid game state)
    if not legal_moves:
        return [0, 1, 2, 3]  # Return all moves
        
    return legal_moves

# Training loop with visualization
def train_agent(episodes):
    """Train the agent and visualize learning progress"""
    start_time = time.time()
    
    # Metrics
    scores = []
    max_tiles = []
    moves_per_game = []
    
    # Monitoring variables for learning
    eval_interval = 10  # Evaluate progress every X episodes
    eval_episodes = 5   # Number of episodes for evaluation
    
    # Training loop
    for episode in tqdm(range(episodes)):
        # Create a new game
        game = Game(create_board(BOARD_SIZE))
        game.add_new_tile(2)
        
        # Track state
        move_count = 0
        prev_board = None
        episode_reward = 0
        
        # Play until game over
        while not game.finished:
            # Get current state
            state = game.board.copy()
            
            # Get legal moves
            legal_moves = get_legal_moves(game, prev_board)
            
            # Get action from agent
            action = agent.act(state, legal_moves)
            
            # Apply the action
            prev_score = game.score
            prev_board = game.board.copy()
            game.move(moves_list[action])
            
            # Calculate reward
            score_increase = game.score - prev_score
            reward = effective_reward_function(game.board, prev_board, score_increase)
            episode_reward += reward
            
            # Check if game over after move
            done = game.finished
            next_state = game.board.copy()
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Increase move counter
            move_count += 1
            
            # Train after collecting some experience
            if move_count % 4 == 0 and len(agent.memory) >= agent.batch_size:
                loss = agent.replay()
                
            # Limit very long games
            if move_count > 1000:
                break
        
        # Final training after episode ends
        if len(agent.memory) >= agent.batch_size:
            agent.replay()
        
        # Record metrics
        scores.append(game.score)
        max_tiles.append(np.max(game.board))
        moves_per_game.append(move_count)
        
        # Print progress
        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            elapsed = time.time() - start_time
            
            # Get Q-values for a standard state to monitor learning
            test_board = np.array([
                [0, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 4, 0],
                [0, 0, 0, 0]
            ])
            q_values, _ = agent.forward(agent.preprocess_state(test_board))
            
            tqdm.write(f"Episode {episode}: Score = {game.score}, Max Tile = {np.max(game.board)}, "
                      f"Moves = {move_count}, Avg Score = {avg_score:.1f}, "
                      f"Epsilon = {agent.epsilon:.3f}, Time = {elapsed:.1f}s")
            tqdm.write(f"Q-values for test state: {q_values.flatten()}")
            
            # Monitor loss if available
            if len(agent.loss_history) > 0:
                # Fix: Convert numpy array to float if needed
                last_loss = agent.loss_history[-1]
                if isinstance(last_loss, np.ndarray):
                    last_loss = float(last_loss)
                tqdm.write(f"Recent loss: {last_loss:.4f}")
                
            # Save model occasionally
            if episode % 100 == 0 and episode > 0:
                agent.save_model(f"dqn_model_ep{episode}.npz")
    
    # Calculate end time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds")
    
    # Finalize by saving the model
    agent.save_model("dqn_model_final.npz")
    
    return agent, scores, max_tiles, moves_per_game

# Run training
agent, scores, max_tiles, moves_per_game = train_agent(ITER)

# Compute moving averages
window_size = min(50, ITER // 20)
scores_ma = moyenne_mobile_scipy(scores, taille_fenetre=window_size)
max_tiles_ma = moyenne_mobile_scipy(max_tiles, taille_fenetre=window_size)
moves_ma = moyenne_mobile_scipy(moves_per_game, taille_fenetre=window_size)

# Get learning statistics
learning_stats = agent.get_learning_stats()
loss_history = learning_stats['loss_history']
q_history = learning_stats['q_history']
reward_history = learning_stats['reward_history']

# Calculate moving averages for learning metrics
if loss_history:
    # Convert numpy arrays to floats if needed
    loss_history_float = [float(x) if isinstance(x, np.ndarray) else x for x in loss_history]
    loss_ma = moyenne_mobile_scipy(loss_history_float, taille_fenetre=min(50, len(loss_history) // 10))
if q_history:
    # Convert numpy arrays to floats if needed
    q_history_float = [float(x) if isinstance(x, np.ndarray) else x for x in q_history]
    q_ma = moyenne_mobile_scipy(q_history_float, taille_fenetre=min(50, len(q_history) // 10))

# Plot results
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Game score plot
axes[0, 0].plot(scores, 'b', alpha=0.3, label='Score')
axes[0, 0].plot(scores_ma, 'r', label=f'Moving Avg ({window_size} games)')
axes[0, 0].set_title('Game Score')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Score')
axes[0, 0].legend()

# Max tile plot
axes[0, 1].plot(max_tiles, 'g', alpha=0.3, label='Max Tile')
axes[0, 1].plot(max_tiles_ma, 'r', label=f'Moving Avg ({window_size} games)')
axes[0, 1].set_title('Max Tile Per Episode')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Max Tile Value')
axes[0, 1].legend()

# Moves per game plot
axes[1, 0].plot(moves_per_game, 'm', alpha=0.3, label='Moves')
axes[1, 0].plot(moves_ma, 'r', label=f'Moving Avg ({window_size} games)')
axes[1, 0].set_title('Moves Per Episode')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Number of Moves')
axes[1, 0].legend()

# Loss history plot
if loss_history:
    axes[1, 1].plot(loss_history_float, 'y', alpha=0.3, label='Loss')
    if len(loss_history) > window_size:
        axes[1, 1].plot(loss_ma, 'r', label=f'Moving Avg')
    axes[1, 1].set_title('Training Loss')
    axes[1, 1].set_xlabel('Training Iteration')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()

# Q-value history plot
if q_history:
    axes[2, 0].plot(q_history_float, 'c', alpha=0.3, label='Avg Q-Value')
    if len(q_history) > window_size:
        axes[2, 0].plot(q_ma, 'r', label=f'Moving Avg')
    axes[2, 0].set_title('Average Q-Values')
    axes[2, 0].set_xlabel('Training Iteration')
    axes[2, 0].set_ylabel('Q-Value')
    axes[2, 0].legend()

# Epsilon decay plot
epsilon_values = [agent.epsilon_min + (1.0 - agent.epsilon_min) * (agent.epsilon_decay ** i) for i in range(ITER)]
axes[2, 1].plot(epsilon_values, 'r', label='Epsilon')
axes[2, 1].set_title('Exploration Rate (Epsilon)')
axes[2, 1].set_xlabel('Episode')
axes[2, 1].set_ylabel('Epsilon')
axes[2, 1].legend()

plt.tight_layout()
plt.savefig('dqn_training_results.png')
plt.show()

# Play evaluation games with the trained agent
def evaluate_agent(agent, num_games=5):
    """Evaluate the trained agent"""
    print("\nEvaluating trained agent...")
    
    eval_scores = []
    eval_max_tiles = []
    eval_moves = []
    
    for i in range(num_games):
        game = Game(create_board(BOARD_SIZE))
        game.add_new_tile(2)
        
        move_count = 0
        prev_board = None
        
        while not game.finished and move_count < 1000:
            state = game.board.copy()
            legal_moves = get_legal_moves(game, prev_board)
            
            # Use agent with no exploration
            agent.epsilon = 0  # No exploration during evaluation
            action = agent.act(state, legal_moves)
            
            prev_board = game.board.copy()
            game.move(moves_list[action])
            move_count += 1
        
        eval_scores.append(game.score)
        eval_max_tiles.append(np.max(game.board))
        eval_moves.append(move_count)
        
        print(f"Game {i+1}: Score = {game.score}, Max Tile = {np.max(game.board)}, Moves = {move_count}")
    
    print(f"\nEvaluation Results:")
    print(f"Average Score: {np.mean(eval_scores):.1f}")
    print(f"Average Max Tile: {np.mean(eval_max_tiles):.1f}")
    print(f"Average Moves: {np.mean(eval_moves):.1f}")

# Run evaluation
evaluate_agent(agent)