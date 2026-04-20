from mola import Matrix
from mola import utils

def test_matrix_variance():
    """Test the computation of the variance of a matrix."""
    x_row = Matrix([1, 2, 4, 7])
    x_col = Matrix([[1], [2], [4], [7]])
    var_row = utils.var(x_row)
    var_col = utils.var(x_col)
    assert(abs(var_row-7) < 1e-12 and abs(var_col-7) < 1e-12)

def test_matrix_std():
    """Test the computation of the standard deviation of a matrix."""
    x_row = Matrix([1, 2, 4, 7])
    x_col = Matrix([[1], [2], [4], [7]])
    std_row = utils.std(x_row)
    std_col = utils.std(x_col)
    assert(abs(std_row-2.64575131106) < 1e-12 and abs(std_col-2.64575131106) < 1e-12)