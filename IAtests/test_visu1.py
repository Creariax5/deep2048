from perceptron.perceptron1957 import Perceptron1957

neurone1 = Perceptron1957(-2, 4)
neurone2 = Perceptron1957(-3, -2)
neurone3 = Perceptron1957(1, -1)

def get_res(x, y):
    res1 = neurone1.run(x, y)
    res2 = neurone2.run(x, y)
    return neurone3.run(res1, res2)

tab = []
size = 10

for i in range(-size, size + 1):
    line = []
    for j in range(-size, size + 1):
        res = get_res(i, j)
        line.append(res)
    tab.append(line)

for line in tab:
    print(line)
