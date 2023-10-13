from mola.matrix import Matrix
from mola.utils import identity, ones
from copy import deepcopy

def linear_least_squares(H,z,W=None):
    """
    Return the parameters of a first-order polynomial in a tuple.
    The parameters are the slope (first element) and the intercept (second element).
    
    Arguments:
    H -- Matrix: the observation matrix of the linear system of equations
    z -- Matrix: the measured values depicting the right side of the linear system of equations
    W -- Matrix: a weight matrix containing the weights for observations in its diagonals
    
    If no 'W' is given, an identity matrix is assumed and all observations are equally weighted.
    """
    if W is None:
        W = identity(H.get_height())
    th = ((H.get_transpose())*W*H).get_inverse() * H.get_transpose() * W * z
    th_tuple = (th.get(0,0), th.get(1,0))
    return th_tuple


def fit_univariate_polynomial(independent_values, dependent_values, degrees=[1], intercept=True, weights = None, regularization_coefficient = None):
    """
    Return the parameters of an nth-order polynomial in a tuple.
    The algorithm uses least squares regression to minimize the term ||y-H*theta||^2, where y is the vector of dependent values, H is the observation matrix, and theta is the vector of parameters.
    The parameters are the coefficients of the polynomial function.
    Optional arguments allow including intercept in the parameters, weighting certain data points over others, and L2 (Tikhonov) regularization.
    If weights and a regularization coefficient are given, the least squares algorithm instead minifies the loss function ||W^(1/2)(y-H*theta)||^2 + ||I*a*theta||^2, where W is a weight matrix, I is an identity matrix and a is the regularization coefficient.
    
    Arguments:
    independent_values -- Matrix: the matrix of independent values
    dependent_values -- Matrix: the matrix of dependent values
    degrees -- a list of degrees of the polynomial terms in the polynomial function that is fitted
    intercept -- Boolean: whether an intercept term should be included in the polynomial function
    weights -- Matrix: an optional weights matrix to weight certain data points over others
    regularization_coefficient -- float: a regularization parameter that is scalar multiplied with the identity matrix
    """
    
    # first, construct the observation matrix H from the independent values
    H = deepcopy(independent_values)
    for col in range(1,len(degrees)):
        H.append_column(independent_values)

    # second, raise the elements in the columns of H to powers corresponding to the user-given degrees
    for col in range(0,len(degrees)):
        current_pow = degrees[col]
        for row in range(H.get_height()):
            H.set(row,col,pow(H.get(row,col),current_pow))

    # third, include intercept if it is desired
    if intercept:
        H.append_column(ones(H.get_height(),1))
    
    if weights is None and regularization_coefficient is None: # simplest case where there is no weights or regularization
        th = ((H.get_transpose())*H).get_inverse() * H.get_transpose() * dependent_values
    elif weights is None and (isinstance(regularization_coefficient,int) or isinstance(regularization_coefficient,float)): # otherwise, if there is no weights but there is regularization
        th = ((H.get_transpose())*H + (identity(H.get_height()))*regularization_coefficient).get_inverse() * H.get_transpose() * dependent_values
    elif regularization_coefficient is None and isinstance(weights,Matrix): # otherwise, if there is no regularization but there is weights
        th = ((H.get_transpose())*weights*H).get_inverse() * H.get_transpose() * weights * dependent_values
    else:
        raise Exception("undefined case in fit_univariate_polynomial()")

    th_tuple = tuple(th.get_column(0))
    return th_tuple




