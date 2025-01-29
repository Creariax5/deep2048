from perceptronSimple import PerceptronSimple
import random
from my_pil import pil_visu

class Network:
    def __init__(self, weightRange=1, inp=2, out=1, length=10, width=10):
        self.weightRange = weightRange
        self.inp = inp
        self.out = out
        self.length = length
        self.width = width

        self.inp_neurones = self.create_layer(inp, self.width)
        self.neurones = self.create_neuron_network()
        self.out_neurones = self.create_layer(width, 3)

        pil_visu(self.evaluate, "network.png")
    
    def create_single_neuron(self, inp):
        w = []
        for i in range(inp):
            w.append(random.randint(-self.weightRange, self.weightRange))
        b = random.randint(-self.weightRange, self.weightRange)
        neurone = PerceptronSimple(w, b)
        return neurone

    def create_layer(self, inp, width):
        layer = []
        for i in range(width):
            neurone = self.create_single_neuron(inp)
            layer.append(neurone)
        return layer
    
    def create_neuron_network(self):
        neurones = []
        for i in range(self.length):
            layer = self.create_layer(self.width, self.width)
            neurones.append(layer)
        return neurones

    def evaluate_layer(self, inp, layer):
        res = []
        for neuron in layer:
            res.append(neuron.forward_propagation(inp))
        return res

    def evaluate(self, inp):
        res = self.evaluate_layer(inp, self.inp_neurones)
        for layer in self.neurones:
            res = self.evaluate_layer(res, layer)
        return self.evaluate_layer(res, self.out_neurones)
