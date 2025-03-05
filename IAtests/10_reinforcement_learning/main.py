from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import my2048
from networkMatrix import NetworkMatrix

def create_model(X, y, IA: NetworkMatrix, nb_iter = 100):
    X, y = X.T, y.T
    Loss, Accuracy = IA.train(X, y, nb_iter)
    
    plt.plot(np.array(Loss) / max(Loss))
    plt.plot(np.array(Accuracy))
    plt.show()
    IA.save()
    return IA

iter = 100
board_size = 4

#IA
width = 4
length = 2
learning_rate = 0.01

IA = NetworkMatrix(nb_input=board_size*board_size, width=int(width), nb_output=4, length=length, learning_rate=0.05)
# IA = create_model(train_data, train_labels, IA, nb_iter=int(4000))

def get_best_move(board):
    X = np.array(board).flatten()
    activations = IA.forward_propagation(X)
    moves_prob = activations[-1].T[-1]
    return np.argmax(moves_prob)


# Collect stats
boards = []
durations = []
moves = []
scores = []
moyenne = 0

Loss = []

for _ in tqdm(range(iter)):
    stats = my2048.play_game(board_size, get_best_move)
    durations.append(stats['duration'])
    moves.append(stats['moves'])
    moyenne += stats['score']
    scores.append(stats['score'])
    boards = stats['boards']

moyenne = moyenne / iter
print("score moyen: ", moyenne)

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,8))
ax1.plot(scores,'b',label='Score moyen')
ax1.set_title('Score Moyen')
ax2.plot([l*learning_rate for l in Loss],'r',label='Loss')
ax2.set_title('Loss')
plt.tight_layout();plt.show()
