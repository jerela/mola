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
        
def identity(dimension):
    identity_matrix = Matrix(dimension,dimension)
    return identity_matrix.make_identity
