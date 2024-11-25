import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def plot_data(data, save_name='plot.png', linear_func=None):
    plt.figure(figsize=(10, 6))
    for class_value in data.iloc[:,-1].unique(): 
        mask = data.iloc[:,-1] == class_value
        plt.scatter(data[mask].iloc[:,0], data[mask].iloc[:,1], label=f'Class {class_value}')
    
    if linear_func:  # Add linear regression line if function provided
        x = data.iloc[:,0]
        y = data.iloc[:,1]
        m, b = linear_func(x, y)
        plt.plot(x, m*x + b, 'r--', label=f'y = {m:.2f}x + {b:.2f}')
    
    plt.xlabel(data.columns[0]), plt.ylabel(data.columns[1])
    plt.legend(), plt.grid(True)
    plt.savefig(save_name)
    plt.close()


def create_pair_plot(data, class_column):
    # Get feature columns (excluding the class column)
    features = [col for col in data.columns if col != class_column]
    
    # Number of features
    n = len(features)
    
    # Create figure and axes
    fig, axes = plt.subplots(n, n, figsize=(15, 15))
    fig.suptitle('Iris Dataset Pair Plot', y=1.02)
    
    # Get unique classes and assign colors
    classes = data[class_column].unique()
    colors = ['red', 'green', 'blue']  # One color per class
    
    # Create scatter plots
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            
            # Plot each class
            for class_idx, class_name in enumerate(classes):
                class_data = data[data[class_column] == class_name]
                
                if i != j:  # Scatter plot
                    ax.scatter(class_data[features[j]], 
                             class_data[features[i]], 
                             c=colors[class_idx], 
                             label=class_name, 
                             alpha=0.5,
                             s=20)
                else:  # Histogram on diagonal
                    ax.hist(class_data[features[i]], 
                          bins=20, 
                          color=colors[class_idx], 
                          alpha=0.5)
            
            # Set labels
            if i == n-1:  # Bottom row
                ax.set_xlabel(features[j])
            if j == 0:    # Leftmost column
                ax.set_ylabel(features[i])
            
            # Remove upper right legend duplicates
            if i == 0 and j == n-1:
                ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig('iris_pair_plot.png')
    plt.close()

def create_kde_plot(data, x_column, y_column):
    # Extract the data
    x = data[x_column]
    y = data[y_column]
    
    # Create grid of points
    xmin, xmax = x.min() - 0.5, x.max() + 0.5
    ymin, ymax = y.min() - 0.5, y.max() + 0.5
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Calculate kernel density
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot density with custom colormap
    im = ax.imshow(np.rot90(f), cmap='plasma', extent=[xmin, xmax, ymin, ymax])
    
    # Add contour lines
    ax.contour(xx, yy, f, colors='white', alpha=0.5)
    
    # Customize plot
    plt.colorbar(im, label='Density')
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f'KDE Plot for Iris-setosa\n{x_column} vs {y_column}')
    
    # Save plot
    plt.savefig('iris_setosa_kde.png')
    plt.close()
