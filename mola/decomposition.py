
from pickletools import read_stringnl_noescape_pair
from mola.matrix import Matrix
from mola.utils import identity, ones

# QR decomposition using Householder reflections
def qrd(A_original):
    """
    Return a two-element tuple of matrices.
    The elements of the tuple are the Q and R matrices from the QR decomposition of the input matrix.
    The original input matrix is decomposed into a rotation matrix Q and an upper triangular matrix R.
    The decomposition is valid for any real square matrix.
    
    Raises an exception if the matrix is not square.
    """
    
    if not A_original.is_square():
        raise Exception("Cannot perform QR decomposition on the matrix because it is not square!")
    
    A = A_original
    rows = A.get_height()
    cols = A.get_width()
    m = max(rows,cols)
    I = identity(m)
    Q = I

    # iterate through the process of setting the i'th row and column to proper values
    for k in range(0,rows-1):
        # construct basis vector
        e1 = identity(rows-k,1)
        
        # get the column of the current submatrix (if k==0, the first column of the full matrix)
        x = A[k:rows,k]

        # calculate rotation Q
        u = x-x.norm_Euclidean()*e1
        v = u*(1./u.norm_Euclidean())
        Q_i = identity(m-k) - 2*v*v.get_transpose()

        # if we are operating on submatrices (always after the first iteration), update Q back to the size of the original input matrix
        if k > 0:
            # create an identity matrix of the original input matrix's dimensions, then overwrite a part of it with the current Q_i submatrix
            I = identity(m)
            for i in range(k,m):
                I[i,i] = 0
            I[k:rows,k:cols] = Q_i
            Q_i = I

        
        # update relevant elements of A and Q
        A = Q_i*A        
        Q = Q_i*Q
    
    # return Q and R, where Q is the full rotation matrix (constructed from Householder reflections) to turn the original input matrix A into a upper triangular matrix R
    return (Q.get_transpose(),A)

