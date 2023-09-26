from mola.matrix import Matrix


def read_matrix_from_file(file_name, delimiter = ','):
    """
    Return a matrix constructed from the contents of a text file.
    
    Arguments:
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
        
def identity(rows, cols = None):
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

def ones(height,width):
    """
    Return a matrix where all elements are 1.
    
    Arguments:
    height -- unsigned integer: height of the matrix
    width -- unsigned integer: width of the matrix
    """
    return Matrix(height,width,1)

def zeros(height,width):
    """
    Return a matrix where all elements are 0.
    
    Arguments:
    height -- unsigned integer: height of the matrix
    width -- unsigned integer: width of the matrix
    """
    return Matrix(height,width,0)

def equals_approx(left,right,precision=1e-12):
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
