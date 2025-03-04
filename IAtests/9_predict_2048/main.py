from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import my2048
from IApredict import IApredict
import minimaxAlgorithm
import newMinmax

iter = 1
board_size = 4

#IA prediction
width = 32
length = 2
learning_rate = 0.01
ai_predictor = IApredict(board_size, width, length, learning_rate)


# Collect stats
boards = []
durations = []
moves = []
scores = []
moyenne = 0

Loss = []

for _ in tqdm(range(iter)):
    stats = my2048.play_game(board_size, newMinmax.best_move)
    durations.append(stats['duration'])
    moves.append(stats['moves'])
    moyenne += stats['score']
    scores.append(stats['score'])
    boards = stats['boards']
    L = ai_predictor.train(boards, scores[-1])
    Loss.append(L)
    ai_predictor.IA.save()

moyenne = moyenne / iter
print("score moyen: ", moyenne)

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,8))
ax1.plot(scores,'b',label='Score moyen');ax1.set_title('Score Moyen')
ax2.plot([l*learning_rate for l in Loss],'r',label='Loss');ax2.set_title('Loss')
plt.tight_layout();plt.show()


# Create scatter plot
# plt.scatter(durations, scores, c=scores)
# plt.colorbar(label='Score')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Score')
# plt.show()
