from perceptronSimple import PerceptronSimple
import random
from my_pil import pil_visu
from network import Network


def single_neuron(nb):
    neurone1 = PerceptronSimple([random.randint(-nb, nb), random.randint(-nb, nb)])

    pil_visu(neurone1.forward_propagation, "simple.png")

class Defined_Neuron_Nb:
    def __init__(self, nb):
        neurone1 = PerceptronSimple([random.randint(-nb, nb), random.randint(-nb, nb)])
        neurone2 = PerceptronSimple([random.randint(-nb, nb), random.randint(-nb, nb)])
        neurone3 = PerceptronSimple([random.randint(-nb, nb), random.randint(-nb, nb)])
        self.neurones = [neurone1, neurone2, neurone3]

        pil_visu(self.evaluate, "3simple.png")
    
    def evaluate(self, inp):
        res1 = self.neurones[0].forward_propagation(inp)
        res2 = self.neurones[1].forward_propagation(inp)
        return self.neurones[2].forward_propagation([res1, res2])

nb = 1

single_neuron(nb)
Defined_Neuron_Nb(nb)