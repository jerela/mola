from mola import Matrix
from mola import regression
from mola import utils

def test_linear_regression():
    H = Matrix([[2,1],[4,1],[6,1]])
    y = Matrix([[0],[1],[2]])
    th = regression.linear_least_squares(H,y)
    assert(th[0]-0.5 < 1e-6 and th[1]-1 < 1e-6)

def test_second_order_polynomial_regression():
    independent_values = Matrix([[-1],[0],[2]])
    dependent_values = Matrix([[1],[0],[4]])
    th = regression.fit_univariate_polynomial(independent_values,dependent_values, degrees=[2], intercept=False)
    assert(utils.equals_approx(th[0],1))
    
def test_first_order_polynomial_regression():
    independent_values = Matrix([ [2],[4],[6] ])
    dependent_values = Matrix([[0],[1],[2]])
    assert(utils.equals_approx(regression.fit_univariate_polynomial(independent_values, dependent_values, degrees=[1, 2], intercept=True), (0.5, 0, -1)))
