from mola import Matrix
from mola import regression
from mola import utils
from mola import decomposition


def test_equals_approx():
    assert(utils.equals_approx([1,2],[1,2]))
    assert(utils.equals_approx(3.5,3.5))
    assert(utils.equals_approx((1,2),(1,2)))
    assert(utils.equals_approx(Matrix(1,2),Matrix(1,2)))

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
    mat1 = Matrix(3,5,1)
    mat2 = utils.identity(3,5)
    assert(mat2*mat1==mat1)
    
def test_singularity():
    # create a singular matrix
    sing = Matrix([[4, 2, 0], [2, 1, 0], [5,12,-5]])
    assert(sing.is_invertible() == False)
    assert(sing.is_singular() == True)
    
def test_rank():
    sing = Matrix([[4, 2, 0], [2, 1, 0], [5,12,-5]])
    A = Matrix([[1, 1, 0, 2], [-1, -1, 0, -2]])
    B = Matrix([[5]])
    assert(sing.get_rank() == 2)
    assert(A.get_rank() == 1)
    assert(B.get_rank() == 1)
