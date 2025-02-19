import time
from perceptronSimple import PerceptronSimple
from pil import show
from main import get_dataset

iris = get_dataset()

neurone = PerceptronSimple(1)

def train(iris, neuron: PerceptronSimple):
    train_set = iris.iloc[:-9]

    for i in range(len(train_set)):
        row = train_set.iloc[i]
        x, y = row.values[:2]
        returnVal = neuron.forward_propagation([x])
        neuron.minimization([x], returnVal, y)
    return neuron

def learning_speed_test(neurone, learning_rate, MAX_ITER=100):
    neurone.learning_rate = learning_rate
    i = 0

    while MAX_ITER != i:
        neurone = train(iris, neurone)
        if i%1 == 0:
            show('test.png', neurone)
        i += 1
    return neurone

print("Initial weights:", neurone.w, neurone.b)
        
print("Training perceptron...")
neurone = learning_speed_test(neurone, 0.0001, 1000)
print("Final weights:", neurone.w, neurone.b)
