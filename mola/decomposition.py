
from mola.matrix import Matrix
from mola.utils import identity, ones
import math
from copy import deepcopy
from random import random

# QR decomposition using Householder reflections
def qrd(A_original: Matrix):
    """
    Return a two-element tuple of matrices.
    
    The elements of the tuple are the Q and R matrices from the QR decomposition of the input matrix.
    The original input matrix is decomposed into a rotation matrix Q and an upper triangular matrix R using Householder reflections.
    The decomposition is valid for any real square matrix.
    
    Arguments:
    A_original -- Matrix: the matrix to be decomposed
    
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


def eigend(S: Matrix):
    """
    Return the matrix of eigenvalues E and matrix of eigenvectors V from the eigendecomposition of matrix S.
    
    This implementation uses the Jacobi eigendecomposition algorithm to compute the eigenvalue decomposition.
    
    Arguments:
    S -- Matrix: the matrix whose eigenvalue decomposition is to be calculated
    
    Raises an exception if the matrix is not symmetric (for now).
    """
    
    # first check if S is symmetric
    # TODO: if isn't, convert to symmetric and calculate how its eigenvalues must be converted back to the original matrix
    if not S.is_symmetric():
        raise Exception("Matrix for eigenvalue decomposition is not symmetric!")

    # function to calculate the non-diagonal Frobenius norm of a matrix
    def off(A):
        sum_term = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    sum_term = sum_term + A[i,j]*A[i,j]
        return math.sqrt(sum_term)

    # function to calculate the Frobenius norm of a matrix
    def frobenius(A):
        sum_term = 0
        for i in range(n):
            for j in range(n):
                sum_term = sum_term + abs(A[i,j])*abs(A[i,j])
        return math.sqrt(sum_term)

    # return the i and k indices such that they correspond to the maximum nondiagonal absolute value 
    def choose_ik():
        i = 0
        k = 0
        for x in range(n):
            for y in range(n):
                if x != y and abs(B[x,y]) > abs(B[i,k]):
                    i = x
                    k = y
        return (i,k)

    # maximum number of iterations to perform
    max_it = 100000
    # dimensions of the square matrix
    n = S.get_height()
    # initialize V to the identity matrix; later to hold the eigenvectors
    V = identity(n)
    # initialize the error tolerance to stop looping
    eps = frobenius(S)*1e-3
    B = deepcopy(S)
    iteration = 0
    # main loop for the algortihm
    while (off(B)>eps):
        #print(off(B))
        # choose (i,k) such that |a[i,k]| is maximal but not on diagonal
        i,k = choose_ik()
        #print((i,k))
        #for i in range(n):
        #    for k in range(n):
        #        if i == k:
        #            continue
        # calculate c and s
        if B[i,k] == 0:
            c = 1
            s = 0
        else:
            tau = (B[i,i]-B[k,k])/(2*B[i,k])
            if tau>=0:
                t = 1/(tau+math.sqrt(1+tau*tau))
            else:
                t = -1/(-tau+math.sqrt(1+tau*tau))
            c = 1/math.sqrt(1+t*t)
            s = t*c
        # calculate G
        G = identity(n)
        G[i,i] = c
        G[k,k] = c
        G[i,k] = s
        G[k,i] = -s
        # set B (matrix of eigenvalues)
        B = G.get_transpose()*B*G
        # set V (matrix of eigenvectors)
        V = V*G
        iteration = iteration + 1
        if iteration > max_it:
            print("Jacobi eigendecomposition algorithm reached maximum number of iterations, breaking")
            break
    
    # sort according to descending absolute eigenvalue
    eigenvalues = B.get_diagonal_elements()
    eigenvalues_abs = (abs(x) for x in eigenvalues)
    order = [x for x in range(n)]
    # get the order of sorted indices
    zipped = zip(order,eigenvalues_abs)
    order_sorted = sorted(zipped,key=lambda x: x[1], reverse=True)
    new_order, eigenvalues_sorted = zip(*order_sorted)

    # construct the sorted list of eigenvalues and the sorted matrix of eigenvectors
    eigenvalues_sorted = []
    eigenvectors_sorted = deepcopy(V)
    for i in range(len(new_order)):
        eigenvalues_sorted.append(eigenvalues[new_order[i]])
        eigenvectors_sorted[:,i] = V[:,new_order[i]]

    
    
    return (eigenvalues_sorted,eigenvectors_sorted)
    

def eigenvector(A: Matrix) -> tuple:
    """
    Return the dominant eigenvector and corresponding eigenvalue of matrix A.
    """
    v = power_method(A)
    e = rayleigh_quotient(A,v)
    return (v,e)

def rayleigh_quotient(A: Matrix, v: Matrix):
    """
    Return the Rayleigh quotient of matrix A.
    The Rayleigh quotient in this case is the eigenvalue corresponding to the eigenvector v.
    
    Arguments:
    A -- Matrix: the matrix whose Rayleigh quotient is to be calculated
    v -- Matrix: the eigenvector corresponding to the eigenvalue to be calculated
    """
    return v.get_conjugate_transpose()*A*v / (v.get_conjugate_transpose()*v)
    
def power_method(A: Matrix) -> Matrix:
    """
    Return the dominant eigenvector of the matrix A.
    
    Arguments:
    A -- Matrix: the matrix whose dominant eigenvector is to be calculated
    """
        
    # initialize the vector with random values using list comprehension
    vector = Matrix([random() for x in range(A.get_height())])
        
    # do the iteration to (hopefully) converge the vector towards its dominant eigenvector
    for i in range(100):
        vector = (A*vector) / ((A*vector).norm_Euclidean())

    return vector


    
