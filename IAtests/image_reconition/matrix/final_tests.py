import pil
import test_network
import sk_dataset

X, target = sk_dataset.get_dataset()
IA = test_network.create_model(X, target.reshape(-1, 1), width=int(16), length=8, learning_rate = 0.2, nb_iter=20000)

IA.load()
pil.show("newData.png", IA, X, target)
