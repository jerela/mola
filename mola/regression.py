from itertools import takewhile
from mola.matrix import Matrix
from mola.utils import identity, ones

def linear_least_squares(H,z,W=None):
    """
    Returns the parameters of a first-order polynomial in a tuple.
    The parameters are the slope (first element) and the intercept (second element).
    Argument 'H' is the observation matrix of the linear system of equations.
    Argument 'z' are the measured values depicting the right side of the linear system of equations.
    Argument 'W' is a weight matrix containing the weights for observations in its diagonals.
    If no weight matrix is given, an identity matrix is assumed and all observations are equally weighted.
    """
    if W is None:
        W = identity(H.get_height())
    th = ((H.get_transpose())*W*H).get_inverse() * H.get_transpose() * W * z
    th_tuple = (th.get(0,0), th.get(1,0))
    return th_tuple


def fit_univariate_polynomial(independent_values, dependent_values, degrees=[1], intercept=True, weights = None):
    """
    Returns the parameters of an nth-order polynomial in a tuple.
    Argument 'independent_values' is the matrix of independent values.
    Argument 'dependent_values' is the matrix of dependent values.
    Argument 'degrees' is a list of degrees of polynomial terms that you want to include in the polynomial function that is fitted.
    Argument 'intercept' is a boolean telling if an intercept term should be included.
    Argument 'weights' is an optional weights matrix to weight certain data points over others.
    """
    
    # first, construct the observation matrix H from the independent values
    H = independent_values
    for col in range(1,len(degrees)):
        H.append_column(independent_values)
        
    # second, raise the elements in the columns of H to powers corresponding to the user-given degrees
    for col in range(1,len(degrees)):
        current_pow = degrees[col]
        for row in range(H.get_height()):
            H.set(row,col,pow(H.get(row,col),current_pow))

    # third, include intercept if it is desired
    if intercept:
        H.append_column(ones(H.get_height(),1))

    # fourth, construct a default weights matrix if none is given
    if weights is None:
        weights = identity(H.get_height())
    
    th = ((H.get_transpose())*weights*H).get_inverse() * H.get_transpose() * weights * dependent_values

    th_tuple = tuple(th.get_column(0))
    return th_tuple



