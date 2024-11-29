# Perceptron 1957:
Entraîner un neurone artificiel sur des données de références (x, y)
pour que celui-ci renforce ses paramètres W à chaque fois qu'une entrée X
est activée en même temps que la sortie y présente dans ces données.

![[Pasted image 20241124185722.png]]

## Comment je l'ai codé:
- [[forward_propagation]]:
```python
def forward_propagation(self, x1, x2):
	y = x1 * self.w1 + x2 * self.w2
	return self.activation(y)
```
- [[fonction d'Activation]]:
```python
def activation(self, y):
	if y < 0:
		return 0
	else:
		return 1
```
- [[function de Minimisation]] des erreurs:
```python
def learning(self, x1, x2, predicted, target):
	self.w1 = self.w1 + self.learning_rate * (target - predicted) * x1
	self.w2 = self.w2 + self.learning_rate * (target - predicted) * x2
	return self.w1, self.w2
```

# Perceptron Simple

- Fonction Sigmoide
- Erreur Logistique ([[Cost Function]])
- Maximum de Vraisemblance
- [[Descente de Gradient]]

## Comment je l'ai codé:
- [[fonction d'Activation]]:
```python
def activation(self, x):
	return 1 / (1 + math.exp(-x))
```
