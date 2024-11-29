import math
import os
import random
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from test_network import create_model
from networkMatrix import NetworkMatrix

size = 64
mode = 'RGB'
mode_size = len(mode)

def show_activation(A):
    features, samples = A[0].shape
    for i in range(3):
        rnd = random.randint(0, samples - 1)
        A1 = A[0]
        features, samples = A1.shape
        
        # This is an image layer
        height = width = size
        channels = mode_size

        activation_img = A1[:, rnd].reshape(height, width, channels)
            
        plt.figure(figsize=(5, 5))
        plt.imshow(activation_img)
        plt.title(f"Image Layer - Sample {i+1}")
        plt.axis('on')
        plt.show()

        A2 = A[1]
        features, samples = A2.shape

        # Find dimensions that make a rectangle close to square
        side_length = int(np.sqrt(features))
        # Calculate height to fit all features
        height = features // side_length
        if features % side_length != 0:
            height += 1
                
        # Create grid of activations
        activation_data = np.zeros(side_length * height)
        activation_data[:features] = A2[:, rnd]
        activation_2d = activation_data.reshape(height, side_length)
            
        plt.figure(figsize=(5, 5))
        plt.imshow(activation_2d, cmap='viridis')
        plt.colorbar()
        plt.title(f"Hidden Layer - {features} neurons (Sample {i+1})")
        plt.axis('on')
        plt.show()

def show_img(X, y):
    if X.shape[0] == size * size * mode_size:  # If X is in (features, samples) format
        X = X.T  # Transform to (samples, features)
    
    if len(X.shape) == 2:
        # Reshape from (samples, features) to (samples, height, width, channels)
        X = X.reshape(-1, size, size, mode_size)
    
    for i in range(3):
        rnd = random.randint(0, len(X) - 1)
        plt.figure(figsize=(5, 5))
        plt.imshow(X[rnd])
        plt.title(f"Label: {'Tom' if y[rnd]==0 else 'Jerry'}")
        plt.axis('off')
        plt.show()

def load_character_images(character_path, label, size):
    X, y = [], []
    
    for img_path in character_path.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Open and convert to grayscale using 'L' mode
            img = Image.open(img_path).convert(mode)
            # Resize to a standard size
            img = img.resize((size, size))
            # Convert to numpy array and normalize
            img_array = np.array(img) / 255.0
            # Reshape to maintain 3D array structure with single channel
            img_array = img_array.reshape(size, size, mode_size)
            X.append(img_array)
            y.append(label)
            
    return X, y


def load_test_images(base_path='../../dataset/tom_and_jerry'):
    # Initialize lists for images and labels
    X = []
    y = []
    
    # Load Tom images (label 0)
    tom_X, tom_y = load_character_images(Path(base_path) / 'tests_tom', 0, size)
    X.extend(tom_X)
    y.extend(tom_y)

    # Load Jerry images (label 1)
    jerry_X, jerry_y = load_character_images(Path(base_path) / 'tests_jerry', 1, size)
    X.extend(jerry_X)
    y.extend(jerry_y)
    
    return X, y


def load_images(base_path='../../dataset/tom_and_jerry'):
    # Initialize lists for images and labels
    X = []
    y = []

    # Load Tom images (label 0)
    tom_X, tom_y = load_character_images(Path(base_path) / 'tom', 0, size)
    X.extend(tom_X)
    y.extend(tom_y)

    # Load Jerry images (label 1)
    jerry_X, jerry_y = load_character_images(Path(base_path) / 'jerry', 1, size)
    X.extend(jerry_X)
    y.extend(jerry_y)
    
    return X, y

def lists_to_numpy_arrays(X, y):
    X = np.array(X)
    show_img(X, y)
    X = X.reshape(X.shape[0], size * size * X.shape[3])
    y = np.array(y).reshape(-1, 1)
    return X, y

# Load the images
X, y = load_images()
X, y = lists_to_numpy_arrays(X, y)
# X1, y1 = load_test_images()
# X1, y1 = lists_to_numpy_arrays(X1, y1)

# Print dataset information
print(f"Dataset shape: {X.shape}")
print(f"Number of Tom images: {np.sum(y == 0)}")
print(f"Number of Jerry images: {np.sum(y == 1)}")

def split_data(X, y, train_percentage=0.9):
    # Generate random indices for splitting
    indices = np.random.permutation(len(X))
    split_index = int(train_percentage * len(X))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    # Split data using random indices
    train_data = X[train_indices]
    test_data = X[test_indices]
    train_labels = y[train_indices]
    test_labels = y[test_indices]
    
    return train_data, test_data, train_labels, test_labels

# Usage in your code:
width = 128
length = 16

train_data, test_data, train_labels, test_labels = split_data(X, y, 0.9)
IA = NetworkMatrix(nb_input=train_data.shape[1], width=int(width), nb_output=train_labels.shape[1], length=length, learning_rate=0.08)

IA = create_model(train_data, train_labels.reshape(-1, 1), IA, nb_iter=int(1500))
# IA.load()

test_data, test_labels = test_data.T, test_labels.T
result = IA.forward_propagation(test_data)
print("Shape of result[0]:", result[0].shape)
print("Type of result[0]:", type(result[0]))
show_activation(result)
print(result[len(result) - 1], test_labels)
pred = IA.get_accuracy(result[len(result) - 1], test_labels)
print(pred)
