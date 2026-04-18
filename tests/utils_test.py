from mola import Matrix
from mola import utils
import statistics

def test_matrix_variance():
    """Test the computation of the variance of a matrix."""
    x_row = Matrix([1, 2, 4, 7])
    x_col = Matrix([[1], [2], [4], [7]])
    var_row = utils.var(x_row)
    var_col = utils.var(x_col)
    assert(abs(var_row-statistics.variance(x_row)) < 1e-12 and abs(var_col-statistics.variance(x_col)) < 1e-12)
