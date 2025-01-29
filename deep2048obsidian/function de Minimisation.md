## Perceptron 1957:
```python
def learning(self, x1, x2, predicted, target):
	self.w1 = self.w1 + self.learning_rate * (target - predicted) * x1
	self.w2 = self.w2 + self.learning_rate * (target - predicted) * x2
	return self.w1, self.w2
```
