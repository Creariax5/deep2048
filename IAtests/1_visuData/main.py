import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import visualizing as visu

# Reading the data
iris = pd.read_csv('../dataset/Iris.csv')

# Check first few rows
print("First 2 rows of the dataset:")
print(iris.head(2))
print("\n")

# Dropping the id column
iris.drop('Id', axis=1, inplace=True)
iris.drop('SepalLengthCm', axis=1, inplace=True)
iris.drop('SepalWidthCm', axis=1, inplace=True)

# Check first few rows
print("First 2 rows of data used:")
print(iris.head(2))
print("\n")

# Use the function
visu.create_pair_plot(iris, 'Species')
print("Pair plot has been saved as 'iris_pair_plot.png'")

# Filter for Setosa
setosa = iris[iris['Species'] == 'Iris-setosa']

# Create the plot
visu.create_kde_plot(setosa, 'PetalLengthCm', 'PetalWidthCm')
print("KDE plot has been saved as 'iris_setosa_kde.png'")
