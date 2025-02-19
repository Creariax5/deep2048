import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import my2048
from ReinforcementLearningAI import ReinforcementLearningAI

# Training parameters
iter = 200
board_size = 3
hidden_size = 256  # Increased for more complex patterns
learning_rate = 0.0005  # Reduced for more stable learning

# Initialize RL AI
ai_rl = ReinforcementLearningAI(
    size=board_size,
    hidden_size=hidden_size,
    learning_rate=learning_rate
)

# Lists to store metrics
class Metrics:
    def __init__(self):
        self.scores = []
        self.max_tiles = []
        self.moves_per_game = []
        self.losses = []
        self.moving_avg_scores = []
        self.best_score = 0
        self.best_tile = 0
        
    def update(self, game, moves_made, total_loss):
        self.scores.append(game.score)
        self.max_tiles.append(np.max(game.board))
        self.moves_per_game.append(moves_made)
        self.losses.append(total_loss / moves_made if moves_made > 0 else 0)
        self.moving_avg_scores.append(
            np.mean(self.scores[-min(50, len(self.scores)):])
        )
        
        # Update best metrics
        if game.score > self.best_score:
            self.best_score = game.score
            # Save network on new best score
            ai_rl.save()

metrics = Metrics()

# Training loop with improved monitoring
progress_bar = tqdm(range(iter), desc="Training Progress")
for episode in progress_bar:
    game = my2048.Numpy2048(board_size)
    total_loss = 0
    moves_made = 0
    old_score = 0
    
    # Game loop
    while not game.is_game_over():
        old_board = game.board.copy()
        action = ai_rl.best_move(old_board)
        valid_move = game.move(['up', 'left', 'down', 'right'][action])
        new_board = game.board.copy()
        new_score = game.score
        
        # Train with enhanced information
        loss = ai_rl.train(
            old_board,
            action,
            new_board,
            old_score,
            new_score,
            game.is_game_over()
        )
        
        total_loss += loss if loss else 0
        moves_made += 1
        old_score = new_score
    
    # Update metrics
    metrics.update(game, moves_made, total_loss)
    
    # Update progress bar with key metrics
    progress_bar.set_postfix({
        'Score': game.score,
        'Max Tile': np.max(game.board),
        'Avg Score': metrics.moving_avg_scores[-1],
        'Epsilon': f"{ai_rl.epsilon:.3f}"
    })
    
    # Save network periodically and print statistics
    if episode % 100 == 0:
        ai_rl.save()
        print(f"\nEpisode {episode}:")
        print(f"Score: {game.score}")
        print(f"Max Tile: {np.max(game.board)}")
        print(f"Moves: {moves_made}")
        print(f"Epsilon: {ai_rl.epsilon:.3f}")
        print(f"Average Score (last 50): {np.mean(metrics.scores[-50:]):.1f}")
        print(f"Best Score: {metrics.best_score}")
        print(f"Best Tile: {metrics.best_tile}")

# Enhanced visualization
plt.style.use('seaborn-v0_8')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot scores with confidence interval
ax1.plot(metrics.scores, 'b-', alpha=0.3, label='Score')
ax1.plot(metrics.moving_avg_scores, 'r-', label='Moving Average')
ax1.set_title('Game Scores')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Score')
ax1.legend()
ax1.grid(True)

# Plot max tiles with trend
ax2.plot([np.log2(x) for x in metrics.max_tiles], 'g-', label='Max Tile (log2)')
z = np.polyfit(range(len(metrics.max_tiles)), [np.log2(x) for x in metrics.max_tiles], 1)
p = np.poly1d(z)
ax2.plot(p(range(len(metrics.max_tiles))), 'r--', label='Trend')
ax2.set_title('Maximum Tile Achieved (log2 scale)')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Tile Value (log2)')
ax2.legend()
ax2.grid(True)

# Plot moves per game
ax3.plot(metrics.moves_per_game, 'm-', label='Moves per Game')
ax3.set_title('Moves per Game')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Number of Moves')
ax3.legend()
ax3.grid(True)

# Plot loss with smoothing
window_size = 50
if len(metrics.losses) > window_size:
    smooth_losses = np.convolve(metrics.losses, np.ones(window_size)/window_size, mode='valid')
    ax4.plot(metrics.losses, 'k-', alpha=0.3, label='Loss')
    ax4.plot(range(window_size-1, len(metrics.losses)), smooth_losses, 'r-', label='Smoothed Loss')
else:
    ax4.plot(metrics.losses, 'k-', label='Loss')
ax4.set_title('Training Loss')
ax4.set_xlabel('Episode')
ax4.set_ylabel('Loss')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()

# Print final statistics
print("\nFinal Statistics:")
print(f"Best Score: {metrics.best_score}")
print(f"Best Max Tile: {metrics.best_tile}")
print(f"Average Score (last 100 episodes): {np.mean(metrics.scores[-100:]):.2f}")
print(f"Max Score: {max(metrics.scores)}")
print(f"Average Max Tile (last 100 episodes): {np.mean(metrics.max_tiles[-100:]):.1f}")
print(f"Best Max Tile: {max(metrics.max_tiles)}")
print(f"Average Moves per Game (last 100): {np.mean(metrics.moves_per_game[-100:]):.1f}")
print(f"Final Epsilon: {ai_rl.epsilon:.4f}")
print(f"Memory Size: {len(ai_rl.memory)}")