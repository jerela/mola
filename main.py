from mola import Matrix, LabeledMatrix
from mola import regression
from mola import utils
from mola import decomposition
from mola import clustering

mat2 = Matrix(3,5,1)
mat3 = mat2.get_transpose()

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



independent_values = Matrix([ [2],[4],[6] ])
dependent_values = Matrix([[0],[1],[2]])
th = regression.fit_univariate_polynomial(independent_values, dependent_values, degrees=[1, 2], intercept=True)
assert(utils.equals_approx(th, (0.5, 0, -1)))

mat11 = Matrix([ [12, -51, 4], [6, 167, -68], [-4, 24, -41] ])
print("original matrix:")
mat11.print()
Q, R = decomposition.qrd(mat11)
print("Q:")
Q.print()
print("R:")
R.print()

independent_values = Matrix([[-1],[0],[2]])
dependent_values = Matrix([[1],[0],[4]])
th = regression.fit_univariate_polynomial(independent_values,dependent_values, degrees=[2], intercept=False)
print(th)
assert(utils.equals_approx(th[0],1))

#(mat11.get_dominant_eigenvector()).print()

# create a symmetric matrix
#mat12 = Matrix([[1,2,3,4,5],[2,7,51,-5,21],[3,51,-11,-23,0],[4,-5,-23,4.2,40.4],[5,21,0,40.4,0.5]])
#[eigs, V] = decomposition.eigend(mat12)
#print(eigs)
#V.print()

mat13 = Matrix([[1,0], [1,3]])
#[eigs, V] = decomposition.eigend(mat13)
#print(eigs)
#V.print()
v,e = decomposition.eigenvector(mat13)
v.print()
print(e)


mat1 = Matrix(3,5,1)
mat2 = utils.identity(3,3)
mat1.print()
mat2.print()
mm = mat2*mat1
mm.print()
assert(mat2*mat1==mat1)



# TEST GAUSS-NEWTON ITERATION
h = Matrix([lambda a,x: pow(a,x)])
independents = Matrix([1, 2, 3]).get_transpose()
y = Matrix([2, 4, 8]).get_transpose()
# let J be the Jacobian of h(x)
J = Matrix([lambda a,x: x*pow(a,x-1)])

theta = regression.fit_nonlinear(independents, y, h, J, initial=Matrix([0.5]))
print(theta)


# TEST K-MEANS CLUSTERING
initial_centers = Matrix([[0,0],[20,0]])
symmetric_points = Matrix([[-1,0],[-2,2],[1,1],[20,2],[18,0],[22,-1],[23,-1]])
centers = clustering.find_k_means(data=symmetric_points,num_centers=2,initial_centers=initial_centers)


print(centers)
assert(utils.equals_approx(centers,Matrix([[-0.6667, 1.0],[20.75, 0.0]]),precision = 1e-4))


# TEST SUBTRACTIVE CLUSTERING
density_points = Matrix([[1,2], [0.5,1.5], [0,1], [0,0.5], [0,0], [0,-0.5], [0,-1], [0.5,-1.5], [1,-2], [2,0], [2.5,-0.5], [3,-1], [3,-1.5], [3,-2], [3,-2.5], [3,-3], [2.5,-3.5], [2,-4]])
mountaintops = clustering.find_density_clusters(data=density_points, num_centers=2, beta = 0.25, sigma = 0.25)
print(mountaintops)
