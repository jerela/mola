import random
import statistics
import math
from mola.matrix import Matrix



def randoms(height: int, width: int) -> Matrix:
    """
    Return a matrix where all elements are random numbers between 0 and 1.
    
    Arguments:
    height -- unsigned integer: height of the matrix
    width -- unsigned integer: width of the matrix
    """
    mat = Matrix(height,width)
    for row in range(height):
        for col in range(width):
            mat.set(row,col,random.random())
    return mat

def write_matrix_to_file(matrix: Matrix, file_name: str, delimiter = ',') -> None:
    """
    Write a matrix to a text file.
    
    Arguments:
    matrix -- Matrix: the matrix to write to the file
    file_name -- string: the name of the file to write to
    delimiter -- character: specifies the delimiter that separates data values in the text file (default ,)
    """
    # open file for writing
    file = open(file_name,'w')
    # write matrix to file
    for row in range(matrix.get_height()):
        for col in range(matrix.get_width()):
            # write value of the current element
            file.write(str(matrix.get(row,col)))
            # if we're not at the end of the row, write the delimiter
            if col < matrix.get_width()-1:
                file.write(delimiter)
        # at the end of the row, write a newline character
        file.write('\n')
    file.close()

def read_matrix_from_file(file_name: str, delimiter = ',') -> Matrix:
    """
    Return a matrix constructed from the contents of a text file.
    
    Arguments:
    file_name -- string: the name of the file to read from
    delimiter -- character: specifies the delimiter that separates data values in the text file (default ,)
    
    If no delimiter is given, the file is assumed to be in comma-separated values format.
    """
    # read all lines from file
    file = open(file_name,'r')
    lines = file.readlines()
    file.close
    
    cols = []
    # parse lines for delimiter
    for line in lines:
        # remove newline characters from the end of the line
        line = line.replace('\n','')
        # split text by delimiters
        split_text = line.split(delimiter)
        # convert to floating-point type
        row = list(map(float,split_text))
        cols.append(row)

    return Matrix(cols)
        
def identity(rows: int, cols = None) -> Matrix:
    """
    Return a square identity matrix.
    
    Arguments:
    rows -- unsigned integer: height of the matrix
    cols -- unsigned integer: width of the matrix (default None)
    
    If 'cols' is not specified, the matrix is assumed to have the same number of columns as the number of rows.
    """
    if cols is None:
        cols = rows
    identity_matrix = Matrix(rows,cols)
    identity_matrix.make_identity()
    return identity_matrix

def ones(height: int, width: int) -> Matrix:
    """
    Return a matrix where all elements are 1.
    
    Arguments:
    height -- unsigned integer: height of the matrix
    width -- unsigned integer: width of the matrix
    """
    return Matrix(height,width,1)

def zeros(height: int, width: int) -> Matrix:
    """
    Return a matrix where all elements are 0.
    
    Arguments:
    height -- unsigned integer: height of the matrix
    width -- unsigned integer: width of the matrix
    """
    return Matrix(height,width,0)

def equals_approx(left, right, precision=1e-12) -> bool:
    """Return true if the compared objects are roughly equal elementwise. Otherwise, return false.
    
    Arguments:
    left -- Matrix, list, tuple, or a single value: the object on the left side of the comparison
    right -- Matrix, list, tuple or a single value: the object on the right side of the comparison
    precision -- float: the maximum allowed difference between matching elements (default 1e-12)
    
    Raises an exception if 'left' and 'right' have different dimensions.
    """
    equals = True
    # if both objects are matrices
    if isinstance(left,Matrix) and isinstance(right,Matrix):
        if not (left.get_height() == right.get_height() and left.get_width() == right.get_width()):
            raise Exception("Exception in equals_approx(): matrices have different dimensions")
        for row in range(left.get_height()):
            equals = equals_approx(left.get_row(row),right.get_row(row),precision)
    # otherwise, if both objects are lists or tuples
    elif (isinstance(left,tuple) or isinstance(left,list)) and (isinstance(right,tuple) or isinstance(right,list)):
        if not (len(left) == len(right)):
            raise Exception("Exception in equals_approx(): objects have different lengths")
        for i in range(len(right)):
            if abs(left[i]-right[i]) > precision:
                equals = False
    # otherwise, if both objects are single values
    else:
        if abs(left-right) > precision:
            equals = False
    return equals


# calculate the mean of a matrix
def get_mean(data: Matrix, along='col') -> Matrix:
    """
    Return the mean of a matrix as a single-row or single-column matrix.
    
    Arguments:
    data -- Matrix: the matrix whose mean is to be calculated
    along -- string: the dimension along which the mean is to be calculated (default 'col')
    """
    
    
    if isinstance(data,Matrix):
        operable_data = data.data
    elif isinstance(data,list):
        operable_data = data
    else:
        raise Exception("exception in utils.mean(): unidentified data type")    

    if along=='col':
        operable_data = transpose_list(operable_data)


    return Matrix([sum(row)/len(row) for row in operable_data])
    

# transpose a 2D list
def transpose_list(data: list) -> list:
    """
    Return the transpose of a 2D list.
    
    Arguments:
    data -- list: the 2D list to be transposed
    """
    width = len(data[0])
    height = len(data)
    data_transposed = []
    for col in range(width):
        new_row = []
        for row in range(height):
            new_row.append(data[row][col])
        data_transposed.append(new_row)
    return data_transposed

# return the unique rows of a matrix or a list
def uniques(data):
    """
    Return the unique rows of a 2D matrix or list.
    
    Arguments:
    data -- Matrix or list: the matrix or list whose unique rows are to be returned
    """

    # convert to lists if necessary
    if isinstance(data,Matrix):
        operable_data = data.data
    elif isinstance(data,list):
        operable_data = data
    else:
        raise Exception("exception in utils.uniques(): unidentified data type")
    
    unique_rows = []
    for row in operable_data:
        if row not in unique_rows:
            unique_rows.append(row)
    return unique_rows

# return the euclidean norm of a matrix
def norm(data: Matrix) -> float:
    """
    Return the Euclidean norm of a matrix.
    You could also just call data.norm_Euclidean() directly, but this is a wrapper function for convenience.
    
    Arguments:
    data -- Matrix: the matrix whose Euclidean norm is to be returned
    """
    return data.norm_Euclidean()

# construct a Matrix that represents a column vector from one-dimensional list
def column(data: list) -> Matrix:
    """
    Return a column vector Matrix object constructed from a one-dimensional list.
    This is the same as calling Matrix(data).get_transpose() with a check to make sure the list is one-dimensional.
    
    Arguments:
    data -- list: the 1D list to be used as the data of the matrix

    Raises an exception if the list is multidimensional.
    """
    if isinstance(list[0],list):
        raise Exception("exception in utils.column(): list is multidimensional")

    return Matrix(data).get_transpose()
    
# calculate the variance of a vector of values
def var(X):
    """
    Return the variance of the input.
    """
    return statistics.variance(X)     
    
# calculate the covariance of two vectors
def cov(X,Y = None) -> float:
    """
    Return the covariance of a random variable or between two random variables.
    
    Note that the result is the "sample covariance" that is normalized by N-1 rather than N, where N is the number of samples.
    Use covmat() if you want to calculate the covariance matrix.
    
    Arguments:
    X -- list or Matrix: the values representing samples from the first random variable or column-organized samples from several random variables
    Y -- list or Matrix: the values representing samples from the second random variable (optional)
    """
    
    # if X is a list and Y is not defined, calculate the covariance of X with itself, i.e., variance
    if isinstance(X,list) and Y is None:
        return var(X)
        
    # if X is a matrix and Y is not defined, try to calculate the covariance between the columns of X
    elif isinstance(X,Matrix) and Y is None:        
        n_cols = X.get_width()
        # if X has more than 2 columns, calculate the covariance matrix
        if n_cols > 2:
            return covmat(X)
        # if X has 2 columns, calculate the covariance between them
        if n_cols == 2:
            return cov(X[:,0],X[:,1])
        # if X has 1 column, calculate its covariance with itself (variance)
        if X.get_width() == 1:
            return var(X.get_column(0,as_list=True))
    
    # if X and Y are both lists, calculate their sample covariance
    elif isinstance(X,list) and isinstance(Y,list):
        n = len(X)
        mX = statistics.mean(X)
        mY = statistics.mean(Y)
        c = 0
        for x,y in zip(X,Y):
            c += (x-mX) * (y-mY)
        c = c/(n-1)
        return c
        
    # if X and Y are both matrices but have 1 column each, calculate their covariance
    elif isinstance(X,Matrix) and isinstance(Y,Matrix):
        if X.get_width() == 1 and Y.get_width() == 1:
            return cov(X.get_column(0,as_list=True),Y.get_column(0,as_list=True))
        else:
            raise Exception("Invalid arguments for cov()")
    
    

    

def std(X):
    """
    Return the standard deviation of the input.
    """
    return statistics.stdev(X)

def covmat(X: Matrix) -> Matrix:
    """
    Return the covariance matrix for the input matrix.
    The diagonal of the covariance matrix contains variances.
    Note that the result is the "sample covariance" that is normalized by N-1 rather than N, where N is the number of samples.
    
    Arguments:
    X -- Matrix: a matrix where the vectors are organized into columns
    """
    if isinstance(X,Matrix):

        n = X.get_height()
        m = X.get_width()
        
        c = zeros(m,m)
        
        for i in range(m):
            for j in range(m):
                c[i,j] = cov(X.get_column(i,as_list=True),X.get_column(j,as_list=True))
                
        return c
    else:
        raise Exception("Input to covmat() must be a Matrix object.")
    
    
def pearson(X,Y):
    return cov(X,Y)/(std(X)*std(Y))
    
def diag(x):
    if isinstance(x,Matrix):
        if x.get_height() == 1:
            v = x.get_row(0,as_list = True)
        elif x.get_width() == 1:
            v = x.get_column(0,as_list = True)
        else:
            raise Exception("Input argument to diag() is invalid")
    n = len(v)
    mat = identity(n)
    for i in range(n):
        mat[i,i] = v[i]
    return mat
    
def correlation_matrix(n, x=None, sigma=0.5, delta=1.1, epsilon=1e-3, phi=1.1, process=None):
    """
    Return a square correlation matrix with given assumptions.
    
    Arguments:
    n -- int: number of rows and columns in the matrix to return
    x -- list or Matrix: a vector defining the variance function, such as a vector of residuals (default None)
    sigma -- float: assumed standard deviation of the constant part of model errors (default 0.5)
    delta -- float: shape parameter of the assumed power function of error variance (default 1.1)
    epsilon -- float: a factor used to prevent the power function from approaching zero (default 1e-3)
    phi -- float: the assumed correlation between successive observations when assuming an AR(1) process
    process -- string: 'AR(1)' or None (default None)
    
    Use this function to return a correlation matrix when assuming a non-constant error variance or correlation between residual errors, or a combination of both.
    """
    W = identity(n)
    
    if process == 'AR(1)' and x is None:
        for i in range(n):
            for j in range(n):
                W[i,j] = phi**abs(i-j)
    elif process == 'AR(1)' and x is not None:
        for i in range(n):
            for j in range(n):
                W[i,j] = sigma**2 * abs(x[i]*x[j])**delta * phi**abs(i-j)
    elif process is None and x is not None:
        for i in range(n):
            W[i,i] = sigma**2 * (epsilon+abs(x[i])**delta)**2
    else:
        raise Exception('Invalid parameters in correlation_matrix()')
    
    return W

def rmse(x1,x2):
    """
    Return the root mean square error of two vectors.
    
    Arguments:
    x1 -- list or Matrix: the first vector
    x2 -- list or Matrix: the vector to compare against
    
    Raises an exception if the vectors are not of equal length.
    """
    if isinstance(x1,Matrix) and isinstance(x2,Matrix):
        x1 = x1.get_column(0,as_list=True)
        x2 = x2.get_column(0,as_list=True)
    
    n = len(x1)
    if len(x1)!=len(x2):
        raise Exception('Arguments must be of equal length in rmse()')
    
    return math.sqrt(sum([(x-y)**2 for x,y in zip(x1,x2)])/n)
    