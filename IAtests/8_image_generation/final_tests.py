import random
from matplotlib import pyplot as plt
import numpy as np
import sk_dataset
from networkMatrix import NetworkMatrix
import my_dataset

def add_noise(X, noise_level=0.1):
    X_noisy = X.copy()
    
    img_range = np.max(X) - np.min(X)
    noise_std = noise_level * img_range
    
    noise = np.random.normal(0, noise_std, X.shape)
    X_noisy = X_noisy + noise
    
    X_noisy = np.clip(X_noisy, np.min(X), np.max(X))
    return X_noisy

def show_img(X, y, size=8, mode_size=1):
    if X.shape[0] == size * size * mode_size:  # If X is in (features, samples) format
        X = X.T  # Transform to (samples, features)
    
    if len(X.shape) == 2:
        # Reshape from (samples, features) to (samples, height, width, channels)
        X = X.reshape(-1, size, size, mode_size)
    
    for i in range(1):
        rnd = random.randint(0, len(X) - 1)
        plt.figure(figsize=(5, 5))
        plt.imshow(X[105])
        # plt.title(f"Label: {'Tom' if y[rnd]==0 else 'Jerry'}")
        plt.axis('off')
        plt.show()

def create_model(X, y, IA, nb_iter = 100):
    X, y = X.T, y.T
    Loss, Accuracy = IA.train(X, y, nb_iter)
    
    plt.plot(np.array(Loss) / max(Loss))
    plt.plot(np.array(Accuracy))
    plt.show()
    IA.save()
    return IA

def test_IA(X, y, IA):
    X, y = X.T, y.T
    activations = IA.forward_propagation(X)
    a = IA.get_accuracy(activations[-1], y)
    prediction = sk_dataset.convert_from_one_hot(activations[-1].T)[:10]
    target = sk_dataset.convert_from_one_hot(y.T)[:10]
    print("prediction   : ", prediction)
    print("target       : ", target)
    print("score        : ", a, "%")

width = 128
length = 3

X, y = my_dataset.get_digits()

for i in range(10):
    show_img(X, y, size=8, mode_size=1)
    X = add_noise(X, noise_level=0.1)

train_data, test_data, train_labels, test_labels = sk_dataset.split_data(X, y, 0.9)

IA = NetworkMatrix(nb_input=train_data.shape[1], width=int(width), nb_output=train_labels.shape[1], length=length, learning_rate=0.1)

IA = create_model(train_data, train_labels, IA, nb_iter=int(4000))
# IA.load()
test_IA(test_data, test_labels, IA)
