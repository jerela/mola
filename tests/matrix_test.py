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
    mat2 = Matrix(3,5,1)
    mat3 = mat2.get_transpose()
    mat4 = mat2.matrix_multiplication(mat3)
    mat5 = mat2*mat3
    assert(mat4==mat5)
        
def test_linear_regression():
    H = Matrix([[2,1],[4,1],[6,1]])
    y = Matrix([[0],[1],[2]])
    th = regression.linear_least_squares(H,y)
    assert(th[0]-0.5 < 1e-6 and th[1]-1 < 1e-6)

def test_second_order_polynomial_regression():
    independent_values = Matrix([[-1],[0],[2]])
    dependent_values = Matrix([[2],[0],[8]])
    th = regression.fit_univariate_polynomial(independent_values,dependent_values, degrees=[2], intercept=False)
    assert(utils.equals_approx(th,(2,0)))
    
def test_first_order_polynomial_regression():
    independent_values = Matrix([ [2],[4],[6] ])
    dependent_values = Matrix([[0],[1],[2]])
    assert(utils.equals_approx(regression.fit_univariate_polynomial(independent_values, dependent_values, degrees=[1, 2], intercept=True), (0.5, 0, -1)))
    
def test_qr_decomposition():
    mat = Matrix([ [12, -51, 4], [6, 167, -68], [-4, 24, -41] ])
    Q, R = decomposition.qrd(mat)
    q = Matrix([ [0.8571, -0.3943, 0.3314], [0.4286, 0.9029, -0.0343], [-0.2857, 0.1714, 0.9429]])
    r = Matrix([ [14, 21, -14], [0, 175, -70], [0, 0, -35] ])
    assert(utils.equals_approx(Q,q,1e-4) and utils.utils.equals_approx(R,R,1e-4))