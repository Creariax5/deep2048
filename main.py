from matrix import Matrix

matrix = Matrix()

matrix.display()
vec = matrix.get_rnd_empty_case()
print()
matrix.matrix[vec.x][vec.y] = 2
matrix.display()
