import pil
import test_network
import sk_dataset
from networkMatrix import NetworkMatrix
import load_tom_jerry

X, target = sk_dataset.get_dataset()
# Usage in your code:
width = 128
length = 16

train_data, test_data, train_labels, test_labels = load_tom_jerry.split_data(X, target, 0.9)
train_labels = train_labels.reshape(-1, 1)

IA = NetworkMatrix(nb_input=train_data.shape[1], width=int(width), nb_output=train_labels.shape[1], length=length, learning_rate=0.01)

IA = test_network.create_model(train_data, train_labels.reshape(-1, 1), IA, nb_iter=int(1500))

IA.load()
# pil.show("newData.png", IA, X, target)
