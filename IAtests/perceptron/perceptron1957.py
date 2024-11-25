class Perceptron1957:
    def __init__(self, w1=0, w2=0):
        self.w1 = w1
        self.w2 = w2
        self.learning_rate = 0.01
    
    def run(self, x1, x2):
        y = x1 * self.w1 + x2 * self.w2
        return self.activation(y)

    def activation(self, y):
        if y < 0:
            return 0
        else:
            return 1
    
    def learning(self, x1, x2, predicted, target):
        self.w1 = self.w1 + self.learning_rate * (target - predicted) * x1
        self.w2 = self.w2 + self.learning_rate * (target - predicted) * x2
        return self.w1, self.w2

