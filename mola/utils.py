from mola.matrix import Matrix


def read_matrix_from_file(file_name, delimiter = ','):
    """
    Returns a Matrix object constructed from the contents of a text file.
    Argument 'delimiter' specifies the character that separates data values in the text file.
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
    Returns a square identity matrix.
    Argument 'dimension' is the width and height of the matrix.
    """
    if cols is None:
        cols = rows
    identity_matrix = Matrix(rows,cols)
    identity_matrix.make_identity()
    return identity_matrix

def ones(height,width):
    """
    Returns a matrix of ones.
    Arguments 'height' and 'width' define the width and height of the matrix.
    """
    return Matrix(height,width,1)

def zeros(height,width):
    """
    Returns a matrix of zeros.
    Arguments 'height' and 'width' define the width and height of the matrix.
    """
    return Matrix(height,width,0)
