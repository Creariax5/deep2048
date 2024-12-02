import pil
import test_network
import sk_dataset
from networkMatrix import NetworkMatrix

X, target = sk_dataset.get_dataset()

IA = NetworkMatrix(nb_input=X.shape[1], width=int(32), nb_output=target.shape[1], length=8, learning_rate=1.5)

IA = test_network.create_model(X, target.reshape(-1, 1), IA, nb_iter=int(20000))

# IA.load()
# pil.show("newData.png", IA, X, target)
