from mola.matrix import Matrix
from mola.utils import identity

def linear_least_squares(H,z,W=0):
    """
    Returns the parameters of a first-order polynomial in a tuple.
    The parameters are the slope (first element) and the intercept (second element).
    Argument 'H' is the observation matrix of the linear system of equations.
    Argument 'z' are the measured values depicting the right side of the linear system of equations.
    Argument 'W' is a weight matrix containing the weights for observations in its diagonals.
    If no weight matrix is given, an identity matrix is assumed and all observations are equally weighted.
    """
    if not isinstance(W,Matrix):
        W = identity(H.get_height())
    th = ((H.get_transpose())*W*H).get_inverse() * H.get_transpose() * W * z
    th_tuple = (th.get(0,0), th.get(1,0))
    return th_tuple
