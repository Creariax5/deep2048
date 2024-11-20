class Matrix:
    def __init__(self, size=6):
        self.size = size
        self.matrix = []
        self.create()

    def create(self):
        for i in range(self.size):
            tmp = []
            for j in range(self.size):
                tmp.append(0)
            self.matrix.append(tmp)

    def display(self):
        for i in range(self.size):
            print(self.matrix[i])
