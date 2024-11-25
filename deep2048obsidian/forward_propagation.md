## Perceptron 1957:
```python
def forward_propagation(self, x1, x2):
	y = x1 * self.w1 + x2 * self.w2
	return self.activation(y)
```
