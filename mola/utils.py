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
