import numpy as np
import shelve

class QNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Add second hidden layer for more complex pattern recognition
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((hidden_size, 1))
        self.W3 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros((output_size, 1))

    def leaky_relu(self, x, alpha=0.01):
        """LeakyReLU activation function for better gradient flow"""
        return np.maximum(alpha * x, x)
    
    def leaky_relu_derivative(self, x, alpha=0.01):
        """Derivative of LeakyReLU"""
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx

    def forward(self, x):
        """Forward pass with two hidden layers"""
        # First hidden layer
        self.z1 = np.dot(self.W1, x) + self.b1
        self.a1 = self.leaky_relu(self.z1)
        
        # Second hidden layer
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = self.leaky_relu(self.z2)
        
        # Output layer
        self.z3 = np.dot(self.W3, self.a2) + self.b3
        
        return self.z3

    def backward(self, x, target):
        """Backward pass with gradient clipping for stability"""
        batch_size = x.shape[1]
        
        # Output layer error
        dz3 = self.z3 - target
        dW3 = np.dot(dz3, self.a2.T) / batch_size
        db3 = np.sum(dz3, axis=1, keepdims=True) / batch_size
        
        # Second hidden layer error
        da2 = np.dot(self.W3.T, dz3)
        dz2 = da2 * self.leaky_relu_derivative(self.z2)
        dW2 = np.dot(dz2, self.a1.T) / batch_size
        db2 = np.sum(dz2, axis=1, keepdims=True) / batch_size
        
        # First hidden layer error
        da1 = np.dot(self.W2.T, dz2)
        dz1 = da1 * self.leaky_relu_derivative(self.z1)
        dW1 = np.dot(dz1, x.T) / batch_size
        db1 = np.sum(dz1, axis=1, keepdims=True) / batch_size
        
        # Gradient clipping to prevent exploding gradients
        clip_value = 1.0
        for grad in [dW1, dW2, dW3, db1, db2, db3]:
            np.clip(grad, -clip_value, clip_value, out=grad)
        
        # Apply Adam optimization
        if not hasattr(self, 'step'):
            self.step = 0
            self.m1, self.v1 = 0, 0  # First moment estimates
            self.m2, self.v2 = 0, 0
            self.m3, self.v3 = 0, 0
            self.beta1, self.beta2 = 0.9, 0.999  # Adam hyperparameters
            self.epsilon = 1e-8
        
        self.step += 1
        
        # Update first hidden layer
        self.m1 = self.beta1 * self.m1 + (1 - self.beta1) * dW1
        self.v1 = self.beta2 * self.v1 + (1 - self.beta2) * (dW1 ** 2)
        m1_hat = self.m1 / (1 - self.beta1 ** self.step)
        v1_hat = self.v1 / (1 - self.beta2 ** self.step)
        self.W1 -= self.learning_rate * m1_hat / (np.sqrt(v1_hat) + self.epsilon)
        self.b1 -= self.learning_rate * db1
        
        # Update second hidden layer
        self.m2 = self.beta1 * self.m2 + (1 - self.beta1) * dW2
        self.v2 = self.beta2 * self.v2 + (1 - self.beta2) * (dW2 ** 2)
        m2_hat = self.m2 / (1 - self.beta1 ** self.step)
        v2_hat = self.v2 / (1 - self.beta2 ** self.step)
        self.W2 -= self.learning_rate * m2_hat / (np.sqrt(v2_hat) + self.epsilon)
        self.b2 -= self.learning_rate * db2
        
        # Update output layer
        self.m3 = self.beta1 * self.m3 + (1 - self.beta1) * dW3
        self.v3 = self.beta2 * self.v3 + (1 - self.beta2) * (dW3 ** 2)
        m3_hat = self.m3 / (1 - self.beta1 ** self.step)
        v3_hat = self.v3 / (1 - self.beta2 ** self.step)
        self.W3 -= self.learning_rate * m3_hat / (np.sqrt(v3_hat) + self.epsilon)
        self.b3 -= self.learning_rate * db3
        
        return float(np.mean(dz3 ** 2))

    def predict(self, x):
        """Get Q-values for a state"""
        return self.forward(x)

    def train(self, x, target):
        """Train on a batch of states and target Q-values"""
        self.forward(x)
        return self.backward(x, target)

    def save(self, filename='qnetwork'):
        """Save network weights, biases, and optimizer state"""
        with shelve.open(filename) as db:
            db['W1'] = self.W1
            db['b1'] = self.b1
            db['W2'] = self.W2
            db['b2'] = self.b2
            db['W3'] = self.W3
            db['b3'] = self.b3
            # Save optimizer state if it exists
            if hasattr(self, 'step'):
                db['step'] = self.step
                db['m1'] = self.m1
                db['v1'] = self.v1
                db['m2'] = self.m2
                db['v2'] = self.v2
                db['m3'] = self.m3
                db['v3'] = self.v3

    def load(self, filename='qnetwork'):
        """Load network weights, biases, and optimizer state"""
        try:
            with shelve.open(filename) as db:
                self.W1 = db['W1']
                self.b1 = db['b1']
                self.W2 = db['W2']
                self.b2 = db['b2']
                self.W3 = db['W3']
                self.b3 = db['b3']
                # Load optimizer state if it exists
                if 'step' in db:
                    self.step = db['step']
                    self.m1 = db['m1']
                    self.v1 = db['v1']
                    self.m2 = db['m2']
                    self.v2 = db['v2']
                    self.m3 = db['m3']
                    self.v3 = db['v3']
                    self.beta1, self.beta2 = 0.9, 0.999
                    self.epsilon = 1e-8
            print("Network loaded successfully")
        except:
            print("No saved network found, using initialized weights")