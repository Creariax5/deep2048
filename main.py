from matrix import Matrix

matrix = Matrix(6)
matrix.matrix[4][3] = 2
matrix.matrix[3][3] = 2
matrix.matrix[5][3] = 4
matrix.matrix[4][2] = 2

matrix.display()
print("\n")
matrix.mouv_up()
matrix.display()



