from molalib import Matrix


mat2 = Matrix(3,5,1)
mat3 = mat2.transpose()


mat4 = mat2.matrix_multiplication(mat3)
mat4.print()

mat5 = mat2*mat3
mat5.print()

print("This should be true:" + str(mat4==mat5))

mat6 = Matrix([[2,1,-1],[-3,-1,2],[-2,1,2]])
mat6.print()
inverse_matrix = mat6.get_inverse()

print("inverse:")
inverse_matrix.print()


#mat7 = Matrix(3,3,1)
mat7 = Matrix([[2,1,-1],[-3,-1,2],[-2,1,2]])
#mat7.set(0,0,2)
#mat7.set(0,2,-1)
#mat7.set(1,0,-3)
#mat7.set(1,1,-1)
#mat7.set(1,2,2)
#mat7.set(2,0,-2)
#mat7.set(2,2,2)
mat8 = mat7*inverse_matrix
print("This should be identity:")
mat8.print()

print("mat8 is symmetric:")
print(mat8.is_symmetric())

print("determinant of mat7:")
print(mat7.get_determinant())

print("Frobenius norm of mat7")
print(mat7.get_norm_Frobenius())


# EXAMPLE: linear regression (fit a line y=ax+b to data)

# define the observation matrix
H = Matrix([[2,1],[4,1],[6,1]])
# define the measurements
y = Matrix([[0],[1],[2]])

# solve a and b (which should be 0.5 and -1)
theta = (H.get_transpose()*H).get_inverse() * H.get_transpose() * y
theta.print()