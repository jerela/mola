from molalib import Matrix


mat2 = Matrix(3,5,1)
mat3 = mat2.transpose()


mat4 = mat2.matrix_multiplication(mat3)
mat4.print()

mat5 = mat2*mat3
mat5.print()

print("This should be true:" + str(mat4==mat5))

mat6 = Matrix(3,3,1)
mat6.set(0,0,2)
mat6.set(0,2,-1)
mat6.set(1,0,-3)
mat6.set(1,1,-1)
mat6.set(1,2,2)
mat6.set(2,0,-2)
mat6.set(2,2,2)
mat6.print()
inverse_matrix = mat6.get_inverse()
print("reduced row echeloned:")
mat6.print()

print("inverse:")
inverse_matrix.print()


mat7 = Matrix(3,3,1)
mat7.set(0,0,2)
mat7.set(0,2,-1)
mat7.set(1,0,-3)
mat7.set(1,1,-1)
mat7.set(1,2,2)
mat7.set(2,0,-2)
mat7.set(2,2,2)
mat8 = mat7*inverse_matrix
print("This should be identity:")
mat8.print()

print(mat8.is_symmetric())

print(mat7.get_determinant())

