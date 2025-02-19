import matplotlib.pyplot as plt
import numpy as np

# Create sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label='Line')
plt.scatter(x, y, color='blue')

# Customize the plot
plt.title('Simple Line Graph')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.legend()

plt.show()
