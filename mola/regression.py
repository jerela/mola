from mola.matrix import Matrix
from mola.utils import identity, ones, zeros, randoms
from copy import deepcopy

def linear_least_squares(H: Matrix, z: Matrix, W=None):
    """
    Return the parameters of a first-order polynomial in a tuple.
    The parameters are the slope (first element) and the intercept (second element).
    
    This implementation uses the least squares criterion to find the parameters that minimize ||y-H*theta||^2, where y is the vector of dependent values, H is the observation matrix, and theta is the vector of parameters.
    In common terms, the algorithm finds the function parameters that minimize the squared sum of differences between the observed values and the values given by the function with the estimated parameters for the given independent values.
    
    Arguments:
    H -- Matrix: the observation matrix of the linear system of equations
    z -- Matrix: the observed or dependent values depicting the right side of the linear system of equations
    W -- Matrix: a weight matrix containing the weights for observations in its diagonals
    
    If no 'W' is given, an identity matrix is assumed and all observations are equally weighted.
    """
    if W is None:
        W = identity(H.get_height())
    th = ((H.get_transpose())*W*H).get_inverse() * H.get_transpose() * W * z
    th_tuple = (th.get(0,0), th.get(1,0))
    return th_tuple


def fit_univariate_polynomial(independent_values: Matrix, dependent_values: Matrix, degrees=[1], intercept=True, weights = None, regularization_coefficient = None):
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

# fit nonlinear function parameters using Gauss-Newton iteration
def fit_nonlinear(independent_values: Matrix, dependent_values: Matrix, h: Matrix, J: Matrix, initial=None, max_iters = 100):
    """
    Return the estimated parameters of a nonlinear model using the Gauss-Newton iteration algorithm.
    
    The algorithm uses Gauss-Newton iteration to find the parameters that minimize the least squares criterion ||y-h(theta)||^2, where y is the vector of dependent values, h is the model function, and theta is the vector of the function's parameters. The estimates are improved iteratively by evaluating the gradient of the least squares criterion and using that gradient to update the parameter estimates in small steps. The gradient is approximated by Jacobian matrices.
    
    Arguments:
    independent_values -- Matrix: the matrix of independent values
    dependent_values -- Matrix: the matrix of dependent values
    h -- Matrix: the model function as a lambda function inside a matrix, where the inputs of the function are the list of parameters and the list of independent values from a single sample
    J -- Matrix: the Jacobian matrix of the model function as lambda functions inside a matrix, where each column corresponds to the partial derivative of the function with respect to one parameter
    initial -- Matrix: the initial guess of the parameters (default None, in which case they are randomized)
    max_iters -- int: the maximum number of iterations (default 100)
    """

    # if no initial guess is given, randomize it
    if initial is None:
        initial = randoms(J.get_width(),1)
    theta = initial
    
    # set the step size
    k = 0.05

    # get the number of data points
    n_samples = independent_values.get_height()

    # initialize the matrices H and Jh that will later hold the model function and Jacobian for each sample and latest parameter estimates
    H = zeros(n_samples,h.get_width())
    Jh = zeros(n_samples,J.get_width())
        
    # main loop
    for i in range(max_iters):
        # evaluate the model function and Jacobian for each samples and each unknown parameter using the latest parameter estimates
        for sample in range(n_samples):
            # construct the values of function h and Jacobian Jh for the current sample
            for col in range(h.get_width()):
                H[sample,col] = h.get(0,col)(theta.get_column(0,as_list=True),independent_values.get_row(sample,as_list=True))
            for col in range(J.get_width()):
                Jh[sample,col] = J.get(0,col)(theta.get_column(0,as_list=True),independent_values.get_row(sample,as_list=True))
        # update theta
        theta = theta + (Jh.get_transpose()*Jh).get_inverse() * Jh.get_transpose() * (dependent_values - H)

    return tuple(theta.get_column(0))
