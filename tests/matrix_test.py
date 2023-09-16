from mola import Matrix
from mola import regression
from mola import utils



def test_inverse():
    matrix = Matrix([[2,1,-1],[-3,-1,2],[-2,1,2]])
    inverse = matrix.get_inverse()
    identity_matrix = utils.identity(3)
    product = matrix*inverse
    assert(product==identity_matrix)
        
def test_read_file():
    mat = utils.read_matrix_from_file('data.txt')
    assert(mat==Matrix([ [1,2,3],[4,5,6],[5,6,7] ]))
        
def test_multiplication():
    mat2 = Matrix(3,5,1)
    mat3 = mat2.transpose()
    mat4 = mat2.matrix_multiplication(mat3)
    mat5 = mat2*mat3
    assert(mat4==mat5)
        
def test_regression():
    H = Matrix([[2,1],[4,1],[6,1]])
    y = Matrix([[0],[1],[2]])
    th = regression.linear_least_squares(H,y)
    assert(th[0]-0.5 < 1e-6 and th[1]-1 < 1e-6)
