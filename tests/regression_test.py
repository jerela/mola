from mola import Matrix
from mola import regression
from mola import utils

def test_linear_regression():
    """Test linear regression (fitting of a first-degree polynomial with an intercept)."""
    H = Matrix([[2,1],[4,1],[6,1]])
    y = Matrix([[0],[1],[2]])
    th = regression.linear_least_squares(H,y)
    assert(th[0]-0.5 < 1e-6 and th[1]-1 < 1e-6)

def test_second_order_polynomial_regression():
    """Test second-order polynomial regression (fitting of a second-degree polynomial)."""
    independent_values = Matrix([[-1],[0],[2]])
    dependent_values = Matrix([[1],[0],[4]])
    th = regression.fit_univariate_polynomial(independent_values,dependent_values, degrees=[2], intercept=False)
    assert(utils.equals_approx(th[0],1))
    
def test_first_order_polynomial_regression():
    """Test first-order polynomial regression (fitting of a first-degree polynomial)."""
    independent_values = Matrix([ [2],[4],[6] ])
    dependent_values = Matrix([[0],[1],[2]])
    assert(utils.equals_approx(regression.fit_univariate_polynomial(independent_values, dependent_values, degrees=[1, 2], intercept=True), (0.5, 0, -1)))

def test_nonlinear_regression():
    """Test nonlinear regression with Gauss-Newton iteration."""
    h = Matrix([lambda a,x: pow(a,x)])
    independents = Matrix([1, 2, 3])
    y = Matrix([2, 4, 8])
    # let J be the Jacobian of h(x)
    J = Matrix([lambda a,x: x*pow(a,x-1)])
    # estimate the parameter (the base a of a^x)
    theta = regression.fit_nonlinear(independents, y, h, J, initial=Matrix([0.5]))
    assert(theta[0] == 2)