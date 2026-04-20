from mola import Matrix
from mola import utils

def test_matrix_variance():
    """Test the computation of the variance of a matrix."""
    x_row = Matrix([1, 2, 4, 7])
    x_col = Matrix([[1], [2], [4], [7]])
    var_row = utils.var(x_row)
    var_col = utils.var(x_col)
    assert(utils.equals_approx(var_row,7) and utils.equals_approx(var_col,7))

def test_matrix_std():
    """Test the computation of the standard deviation of a matrix."""
    x_row = Matrix([1, 2, 4, 7])
    x_col = Matrix([[1], [2], [4], [7]])
    std_row = utils.std(x_row)
    std_col = utils.std(x_col)
    assert(utils.equals_approx(std_row,2.6457513110645907) and utils.equals_approx(std_col,2.6457513110645907))
    
def test_matrix_covariance():
    """Test the computation of the covariance of a matrix."""
    x1 = Matrix([1, 2, 4, 7])
    x2 = Matrix([[0, 1, 2, 3], [3, 2, 1, 0]])
    x3 = Matrix([[1, 2, 6], [4, 5, 0], [7, 2, 9]])

    c1 = utils.cov(x1)
    c2 = utils.cov(x2)
    c3 = utils.cov(x3)

    assert(utils.equals_approx(c1,7) and utils.equals_approx(c2,-1.6666666666666667) and utils.equals_approx(c3,Matrix([[9.0, 0.0, 4.5], [0.0, 3.0, -7.5], [4.5, -7.5, 21.0]])))