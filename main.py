from mola import Matrix
from mola import regression
from mola import utils
from mola import decomposition

mat2 = Matrix(3,5,1)
mat3 = mat2.get_transpose()

mat4 = mat2.matrix_multiplication(mat3)
mat4.print()

mat5 = mat2*mat3
mat5.print()

print("This should be true:" + str(mat4==mat5))

mat6 = Matrix([[2,1,-1],[-3,-1,2],[-2,1,2]])
mat6.print()

row_vector = Matrix([1, 0, 0])
row_vector.print()

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


# EXAMPLE: weighted linear regression
# define the observation matrix
H = Matrix([[2,1],[4,1],[6,1]])
# define the measurements
y = Matrix([[0],[1],[2]])

W = Matrix(3,3)
W.make_identity()
W.set(0,0,2) # make the first data sample twice as important as the others

# solve a and b (which should be 0.5 and -1)
theta = (H.get_transpose()*W*H).get_inverse() * H.get_transpose() * W * y
theta.print()

th = regression.linear_least_squares(H,y,W)
print(th)


mat10 = utils.read_matrix_from_file('data.txt')
mat10.print()
H = Matrix([[2,1],[4,1],[6,1]])
y = Matrix([[0],[1],[2]])
th = regression.linear_least_squares(H,y)

independent_values = Matrix([ [2],[4],[6] ])
dependent_values = Matrix([[0],[1],[2]])
print(regression.fit_univariate_polynomial(independent_values, dependent_values, degrees=[1, 2], intercept=True))


mat11 = Matrix([ [12, -51, 4], [6, 167, -68], [-4, 24, -41] ])
print("original matrix:")
mat11.print()
Q, R = decomposition.qrd(mat11)
print("Q:")
Q.print()
print("R:")
R.print()

