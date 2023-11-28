from copy import deepcopy
import math
import random


class Matrix:
    """
    Class that represents a real matrix used in linear algebra tasks. Methods include transpose, inverse, norms, etc. The values of the matrix are stored in a nested list where the outer list contains the rows of the matrix and the inner lists (the rows) contain the elements of the rows. Therefore, when accessing the data, the first index refers to the row and the second index refers to the column.
    
    Attributes:
    data -- nested list: contains the numeric values in the matrix, implemented as a list of lists that represent the rows of the matrix
    n_rows -- unsigned integer: the number of rows in the matrix, also known as its height
    n_cols -- unsigned integer: the number of columns in the matrix, also known as its width
    """
    n_rows = 0
    n_cols = 0
    data = list

    def __init__(self, *args):
        if len(args) == 1:
            self.__construct_from_lists(args[0])
        elif len(args) == 2:
            self.__construct_by_dimensions(args[0], args[1])
        elif len(args) == 3:
            self.__construct_by_dimensions(args[0], args[1], args[2])


    # construct a matrix with r rows, c columns, and some initial value (default 0)
    def __construct_by_dimensions(self,r,c,value=0):
        """
        Return a matrix of specified dimensions. The elements are initialized to the specified value. If no value is provided, all elements are set to zero.
        
        Arguments:
        r -- unsigned integer: the number of rows
        c -- unsigned integer: the number of columns
        value -- float or int: initial value assigned to all elements of the matrix (default 0)
        """
        self.n_rows = r
        self.n_cols = c
        col = []
        for j in range(r):
            row = []
            for i in range(c):
                row.append(value)
            col.append(row)
        self.data = col
    
    # construct a matrix from a given list of lists
    def __construct_from_lists(self,lists):
        """
        Return a matrix constructed from a list.
        
        Arguments:
        lists -- the list or nested list to set as the underlying data of the matrix
        """
        # first check if lists is just an int; with how the class manages constructors, this condition should no longer occur
        if isinstance(lists,int):
            self.n_rows = 1
            self.n_cols = 1
            col = []
            col.append([lists])
            self.data = col
            raise Exception("Input parameter in constructor was an integer but code was executed in condition that should be deprecated in Matrix::construct_from_lists()")
        # check if list is more than 1D (assumedly 2D) by seeing if the first element in the input parameter list is also a list
        elif isinstance(lists[0],list):
            self.n_rows = len(lists)
            self.n_cols = len(lists[0])
            col = []
            for j in range(self.n_rows):
                row = lists[j]
                col.append(row)
            self.data = col
        # if the input parameter was not a nested list, assume it was a simple 1D list and construct a Matrix that is actually a row vector
        elif isinstance(lists,list):
            self.n_rows = 1
            self.n_cols = len(lists)
            self.data = [lists]

    def __abs__(self):
        """
        Return the absolute value of a 1x1 matrix (i.e., a matrix with just one element).
        The returned value is a numeric type, not a matrix.
        
        Raises an exception if the matrix is larger than 1x1.
        """
        if self.n_rows == 1 and self.n_cols == 1:
            return abs(self.data[0][0])
        else:
            raise Exception("Matrix is larger than 1x1 in Matrix::abs()")
        
        
    # overload square brackets ([]) operator
    # first to get data
    def __getitem__(self,idx):
        """
        Return a matrix from the specified indices.
        Overloads the get[] operator.
        
        Arguments:
        idx -- a slice or an integer, or a two-element tuple of slices and/or integers
        
        Raises an exception if 'idx' is not recognized as a type with defined behaviour.
        """
        # first, check if the given index is a slice (several rows as index) or an integer (only one row as the index)
        if isinstance(idx, slice) or isinstance(idx,int):
            # if the index is a slice, return a matrix with the specified row(s)
            return Matrix(self.data[idx])
        # otherwise, check if the given index is a tuple of two slices or integers (i.e., both row(s) and column(s) defined as the index)
        elif isinstance(idx,tuple):
            rows,cols = idx
            # rows is int and cols is slice (several values from a single row are queried)
            if isinstance(rows,int) and isinstance(cols,slice):
                #sliced_data = Matrix(self.data[rows][cols])
                return Matrix(self.data[rows][cols])
            # rows is int and cols is int (one value from given row and column indices is queried)
            elif isinstance(rows,int) and isinstance(cols,int):
                #sliced_data = self.data[rows][cols]
                return self.data[rows][cols]
            # rows is slice and cols is int (several values from a single column are queried)
            elif isinstance(rows,slice) and isinstance(cols,int):
                #sliced_data = Matrix([r[cols] for r in self.data[rows]])
                return Matrix([r[cols] for r in self.data[rows]]).get_transpose()
            # if both are slices (a submatrix is queried)
            elif isinstance(rows,slice) and isinstance(cols,slice):
                #newlist = [r[cols] for r in self.data[rows]]
                #sliced_data = Matrix(newlist)
                return Matrix([r[cols] for r in self.data[rows]])
                
            #return sliced_data

        else:
            raise Exception("invalid getitem arg")

    # then to set data
    def __setitem__(self,idx,value):
        """
        Set elements of the matrix.
        Overloads the set[] operator.
        
        Arguments:
        idx -- a slice or an integer, or a two-element tuple of slices and/or integers
        
        Raises an exception if 'idx' is not recognized as a type with defined behaviour.
        """
        if isinstance(idx,tuple):
            rows, cols = idx

            # if the given indices are integers (not slices) and the given value is also a single numeric type
            if isinstance(rows,int) and isinstance(cols,int) and (isinstance(value,float) or isinstance(value,int)):
                self.data[rows][cols] = value
            # otherwise, if the value is a single numeric type but either of the indices is not an integer (so likely a slice, or perhaps a list)
            elif isinstance(value,float) or isinstance(value,int):
                for r in range(rows):
                    for c in range(cols):
                        self.data[r][c] = value
            # otherwise, if both indices are slices and the given value is a matrix object
            elif isinstance(value,Matrix) and isinstance(rows,slice) and isinstance(cols,slice):
                i = 0
                for r in range(rows.start, rows.stop, 1):
                    j = 0
                    for c in range(cols.start, cols.stop, 1):
                        self.data[r][c] = value.data[i][j]
                        j = j + 1
                    i = i + 1
            # otherwise, if either of the indices is a slice and the given value is a matrix object
            elif isinstance(value,Matrix) and isinstance(rows,int) and isinstance(cols,slice):
                i = 0
                if cols == slice(None,None,None):
                    cols = slice(0,self.n_cols,1)
                for c in range(cols.start, cols.stop, 1):
                    self.data[rows][c] = value.data[i][c]
            else:
                raise Exception("Undefined behaviour in setitem. Needs to be defined.")
        elif isinstance(idx,int):
            self.set_row(idx,value)
        else:
            raise Exception("invalid setitem arg")
        
    # overload equals (==) operator
    def __eq__(self, other):
        """
        Return true if the matrices are equal elementwise. Otherwise, return false.
        Overloads the equality== operator.
        
        Arguments:
        other -- Matrix: right side of the equality
        
        Raises an exception if the dimensions of the matrices to compare do not match.
        """
        # first check that dimensions match; if not, raise exception
        if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
            raise Exception("Matrix dimensions do not match. Left matrix is " + str(self.n_rows) + "x" + str(self.n_cols) + ", right matrix is " + str(other.get_height()) + "x" + str(other.get_width()) + ".")
        
        # assume that the matrices are equal; compare each element and if any exists that isn't equal, change the assumption to false
        equals = True

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.data[i][j] != other.data[i][j]:
                    equals = False
        return equals

    # overload multiplication (*) operator
    def __mul__(self, other):
        """
        Return the matrix product or scalar product of a matrix and the object 'other' multiplied from the right.
        Overloads the multiplication * operator.
        This function doesn't handle the calculations, but rather calls the appropriate algorithm depending on the type of the inputs.
        
        Arguments:
        other -- Matrix or scalar, the term that is multiplied with the matrix
        
        If 'other' is a Matrix, the output is a matrix that is the product of the two matrices.
        If 'other' is an int or a float, the output a matrix whose elements have been multiplied by 'other'.
        
        Raises an exception if 'other' is not Matrix, int or float, or if 'other' is a Matrix but its dimensions are invalid for multiplication with 'self'.
        """
        # if both left and right are matrices, perform matrix multiplication
        if isinstance(self,Matrix) and isinstance(other,Matrix):
            # check if the number of columns of the calling matrix equals the number of rows of the target matrix
            if self.n_cols != other.get_height():
                raise Exception("Cannot perform matrix multiplication because the number of columns in the left matrix doesn't match the number of rows in the right matrix. Left matrix has " + str(self.n_cols) + " columns, right matrix has " + str(other.get_height()) + " rows.")
            # for small matrices, use the naive method for matrix multiplication; otherwise, transpose the target matrix on the right first
            if self.n_cols < 10:
                return self.__matrix_multiplication_naive(other)
            else:
                return self.__matrix_multiplication_transposed(other)
        # otherwise, if left side is a matrix and right side is a scalar, perform scalar multiplication
        elif isinstance(self,Matrix) and isinstance(other,int):
            return self.__scalar_multiplication(other)
        elif isinstance(self,Matrix) and isinstance(other,float):
            return self.__scalar_multiplication(other)
        # otherwise, we the operation may not be valid and we raise an exception
        else:
            print(type(other))
            raise Exception("Cannot identify type of term on right when multiplying in Matrix::mul()")
    
    # enable multiplication from either direction
    def __rmul__(self, other):
        """
        Return the matrix resulting from a scalar product with the argument.
        
        Arguments:
        other -- numeric or int
        
        Raises an exception if 'other' is not scalar or int.
        """
        if isinstance(other,int) or isinstance(other,float):
            return self.__scalar_multiplication(other)
        else:
            raise Exception("Unknown rmul!")

    # overload division (/) operator
    def __truediv__(self,other):
        """
        Return a single numeric value that is the element of the matrix divided by 'other'.
        Defined only for matrices of height 1 and width 1 (single-element matrix).
        Overloads the divison / operator.
        
        Arguments:
        other -- the divisor
        
        Raises an exception if matrix dimensions are not 1x1.
        """
        # if the denominator is a 1x1 matrix, return the element divided by the argument
        if isinstance(other,Matrix):
            if other.n_rows == 1 and other.n_cols == 1:
                return self.data[0][0]/other
        # otherwise, assume that the denominator is a numeric type and call mul()
        else:
            return self.__mul__(1/other)
#            raise Exception("Cannot perform division because matrix dimensions are not 1x1.")
    
    # enable division from either direction
    def __rtruediv__(self,other):
        """
        Return a single numeric value that is the argument 'other' divided by the element of the matrix.
        Defined only for matrices of height 1 and width 1 (single-element matrix).
        Overloads the division / operator.
        
        Arguments:
        other -- the dividend
        
        Raises an exception if matrix dimensions are not 1x1.
        """
        if self.n_rows == 1 and self.n_cols == 1:
            return other/self.data[0][0]
        else:
            raise Exception("Cannot perform division because matrix dimensions are not 1x1.")

    # overload addition (+) operator
    def __add__(self,other):
        """
        Return a Matrix that is the sum of two matrices or the original matrix where a scalar has been added to all elements.
        Overloads the addition + operator.
        
        Arguments:
        other -- the matrix or scalar to add to the matrix on the left
        
        Raises an exception if 'other' is a matrix but its dimensions do not match those of the matrix on the left.
        """
        output = Matrix(self.n_rows,self.n_cols,0)
        # first, ensure that the dimensions of the matrices match
        if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
            raise Exception("Matrix dimensions must match for elementwise addition or subtraction! Left side is ", str(self.n_rows), "x", str(self.n_cols), " and right side is ", str(other.get_height()), "x", str(other.get_width()), "!")
        # if we passed the dimension check, we can add the matrices elementwise
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                output.set(i,j,self[i,j]+other[i,j])
        return output
    
    # overload subtraction (-) operator
    def __sub__(self,other):
        """
        Return a matrix that is the subtraction of two matrices or the original matrix where a scalar has been subtracted from all elements.
        Overloads the subtraction - operator.
        
        Arguments:
        other -- the matrix or scalar to subtract from the matrix on the left
        
        Raises an exception if 'other' is a matrix but its dimensions do not match those of the matrix on the left.
        """
        output = Matrix(self.n_rows,self.n_cols,0)
        # first, ensure that the dimensions of the matrices match
        if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
            raise Exception("Matrix dimensions must match for elementwise addition or subtraction! Left side is " + str(self.n_rows) + "x" + str(self.n_cols) + ", right side is " + str(other.n_rows) + "x" + str(other.n_cols))
        # if we passed the dimension check, we can subtract the matrices elementwise
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                output.set(i,j,self.data[i][j]-other.data[i][j])
        return output
                
    # return the number of rows
    def get_height(self):
        """Return the number of rows in the matrix."""
        return self.n_rows
    
    # return the number of columns
    def get_width(self):
        """Return the number of columns in the matrix."""
        return self.n_cols
    
    # return a row as a list
    def get_row(self, r: int, as_list=True):
        """
        Return a specified row of the matrix as list or matrix.
        
        Arguments:
        r -- unsigned integer: index of the row
        as_list - Boolean: whether to return the row as a list or not, in which case it is returned as a matrix (default true)
        """
        if as_list:
            return self.data[r]
        else:
            return self.construct_from_lists(self.data[r])
    
    def get_column(self, c: int, as_list=True):
        """
        Return a specified column of the matrix as list or matrix.
        
        Arguments:
        c -- unsigned integer: index of the column
        as_list - Boolean: whether to return the column as a list or not, in which case it is returned as a matrix (default true)
        """
        column = []
        for i in range(self.n_rows):
            column.append(self.data[i][c])
        if as_list:
            return column
        else:
            return self.construct_from_lists(column)
    
    # set a row at given index to given values from a list
    def set_row(self, r: int, new_row: list):
        """
        Set the specified row of the matrix.

        Arguments:
        r -- unsigned integer: index of the row
        new_row -- list: the values that are assigned to that row
        """
        self.data[r] = new_row

    # set a single value in a given index
    def set(self, i: int, j: int, value):
        """Set the element at specified position."""
        self.data[i][j] = value

    # get a single value in a given index
    def get(self, i: int, j: int):
        """Get the element at specified position."""
        return self.data[i][j]

    # define what happens when the matrix is converted to a string, such as when print(Matrix) is called
    def __str__(self, precision = 4):
        """
        Return a string that describes the matrix when string() is called with the matrix as the argument.
        
        This function defines the string representation of Matrix.
        Rows are delimited by semicolons and newlines. Elements in a single row are delimited by commas.
        The matrix is enclosed with square brackets.
        """
        matrix_string = '['
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                matrix_string = matrix_string + str(round(self.data[i][j],precision))
                if j < self.n_cols-1:
                    matrix_string = matrix_string + ", "
            if i < self.n_rows-1:
                matrix_string = matrix_string + ";\n"
        matrix_string = matrix_string + "]"
        return matrix_string

    # print matrix in MATLAB-style format
    def print(self, precision = 4):
        """
        Print a string that describes the matrix.
        Rows are delimited by semicolons and newlines. Elements in a single row are delimited by commas.
        The matrix is enclosed with square brackets.
        Calls the overloaded str() function with the possibility to specify how many decimals are shown.
        
        Arguments:
        precision -- unsigned integer: the number of decimals shown in the output (default 4)
        """
        print(self.__str__(precision))

    # check if matrix elements are real
    def is_real(self) -> bool:
        """Return true if all elements of the matrix are real-valued. Otherwise, return false."""
        real = True
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if not isinstance(self.data[i][j],float) and not isinstance(self.data[i][j],int):
                    real = False
        return real

    def is_identity(self) -> bool:
        """Return true if the matrix is an identity matrix. Otherwise, return false."""
        identity = True
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if i == j:
                    if self.data[i][j] != 1:
                        identity = False
                else:
                    if self.data[i][j] != 0:
                        identity = False
        return identity

    def is_square(self) -> bool:
        """Return true if the matrix is square (number of columns equals number of rows). Otherwise, return false."""
        return self.n_rows == self.n_cols

    # a square real matrix is orthogonal if it multiplied by its tranpose is an identity matrix (its tranpose is its inverse)
    def is_orthogonal(self):
        """Return true if the matrix is real and square and its transpose is its inverse. Otherwise, return false."""
        return self.is_real() and self.is_square() and (self*self.get_transpose()).is_identity()

    # get Frobenius norm of matrix
    def get_norm_Frobenius(self):
        """Return the Frobenius norm of the matrix."""
        return math.sqrt((self.get_conjugate_transpose()*self).get_trace())

    # form a conjugate transpose of the matrix
    def get_conjugate_transpose(self):
        """
        Return the conjugate tranpose of the matrix.
        For real matrices, the conjugate transpose is the transpose.
        """
        if self.is_real():
            return self.get_transpose()
        else:
            raise Exception("Function Matrix::get_conjugate_transpose() is not defined for non-real matrices yet!")

    # transpose a matrix
    def transpose(self):
        """Transpose the matrix."""
        transposed = Matrix(self.n_cols,self.n_rows)
        for i in range(self.n_cols):
            for j in range(self.n_rows):
                transposed.set(i,j,self.data[j][i])
        self.data = transposed.data
        self.n_rows,self.n_cols = self.n_cols, self.n_rows
    
    # return the transpose of a matrix
    def get_transpose(self):
        """Return the transpose of the matrix."""
        calling_matrix = deepcopy(self)
        calling_matrix.transpose()
        return calling_matrix

    # return matrix product; this is faster than the transposed version for small matrices
    def __matrix_multiplication_naive(self,target_matrix):
        """Return the matrix product of two matrices.
        
        Arguments:
        target_matrix --- the matrix on the right side of multiplication
        """
        n_rows = self.n_rows
        n_cols = target_matrix.get_width()
        product_matrix = Matrix(n_rows,n_cols)
        length = self.n_cols
        
#        target_transpose = target_matrix.get_transpose()

        for i in range(n_rows):
            current_row = self.get_row(i)
            #current_row_target = target_transpose.get_row(i)
            for j in range(n_cols):
                new_elem = 0
                for x in range(length):
                    new_elem = new_elem + current_row[x]*target_matrix.data[x][j]
                product_matrix.set(i,j,new_elem)
            
        return product_matrix

    # return matrix product where the data of the target matrix is first transposed; this faster than the naive version for large matrices
    def __matrix_multiplication_transposed(self,target_matrix):
        """Return the matrix product of two matrices.
        
        Transposes the data of the matrix on the right before entering the multiplication loop. This improves performance for matrices that aren't very small (width or height is greater than 10) compared to the "naive" approach.
        
        Arguments:
        target_matrix --- the matrix on the right side of multiplication
        """        
        n_rows = self.n_rows
        n_cols = target_matrix.get_width()
        product_matrix = Matrix(n_rows,n_cols)
        length = self.n_cols

        # transpose the data of the target matrix so that its elements can be accessed more efficiently in the multiplication loop
        target_data = target_matrix.data
        target_data_transposed = []
        for col in range(target_matrix.get_width()):
            new_row = []
            for row in range(target_matrix.get_height()):
                new_row.append(target_data[row][col])
            target_data_transposed.append(new_row)
                

#        target_transpose = target_matrix.get_transpose()

        # loop through rows in the calling matrix
        for i in range(n_rows):
            # get current row of the calling matrix so using it is faster later
            current_row = self.get_row(i)
            # loop through the columns of the target matrix
            for j in range(n_cols):
                # get current row of the target matrix so using it is faster in the following loop
                current_row_target = target_data_transposed[j]
                # initialize the sum to zero
                new_elem = 0
                # loop through the elements of the current row of the calling matrix and the current column of the target matrix
                for x in range(length):
                    new_elem = new_elem + current_row[x]*current_row_target[x]
                # set the element of the product matrix
                product_matrix.set(i,j,new_elem)
            
        return product_matrix

    # return matrix product
    def __matrix_multiplication_tiled(self,target_matrix):
        """Return the matrix product of two matrices.
        
        Arguments:
        target_matrix --- the matrix on the right side of multiplication
        """
        n = self.n_rows
        p = target_matrix.get_width()
        product_matrix = Matrix(n,p)
        m = self.n_cols

        # pick tile size
        T = 4
        for I in range(0,n,T):
            for J in range(0,p,T):
                for K in range(0,m,T):
                    #print("I" + str(I) + " J" + str(J) + " K" + str(K))
                    product_matrix[I:min(I+T,n),J:min(J+T,m)] = self[I:min(I+T,n),K:min(K+T,p)].__matrix_multiplication(target_matrix[K:min(K+T,p),J:min(J+T,m)])
                    #for i in range(I,min(I+T,n)):
                    #    for j in range(J,min(J+T,p)):
                    #        summed = 0
                    #        left_row = self.data[i]
                    #        for k in range(K,min(K+T,m)):
                    #            summed = summed + left_row[k]*target_matrix.data[k][j]
                    #        product_matrix[i,j] = product_matrix[i,j] + summed
        return product_matrix

    # return matrix product
    def __matrix_multiplication_dac(self,target_matrix):
        """Return the matrix product of two matrices.
        This implementation uses the divide and conquer algorithm, where the matrices are recursively split into submatrices until the dimension is below a certain threshold and we perform a "normal" matrix multiplication.
        
        Arguments:
        target_matrix --- the matrix on the right side of multiplication
        
        Raises an exception if the number of columns of the calling matrix does not match the number of rows of the target matrix.
        """
        
        # check if the number of columns of the calling matrix equals the number of rows of the target matrix
        if self.n_cols != target_matrix.get_height():
            raise Exception("Cannot perform matrix multiplication because the number of columns in the left matrix doesn't match the number of rows in the right matrix. Left matrix has " + str(self.n_cols) + " columns, right matrix has " + str(target_matrix.get_height()) + " rows.")
        
        n = self.n_rows
        m = self.n_cols
        p = target_matrix.n_cols
        # get maximum of n, m, p
        max_dim = max(n,m,m)

        if max_dim <= 2:
            return self.__matrix_multiplication(target_matrix)
        elif max_dim == n:
            # split the left matrix horizontally
            #A1 = self[0:n//2,0:m]
            #A2 = self[n//2:n,0:m]
            #return (A1*target_matrix).concatenate(A2*target_matrix,dim='vertical')
            return (self[0:n//2,0:m]*target_matrix).concatenate(self[n//2:n,0:m]*target_matrix,dim='vertical')
        elif max_dim == p:
            #split the right matrix vertically
            #B1 = target_matrix[0:m,0:p//2]
            #B2 = target_matrix[0:m,p//2:p]
            #return (self*B1).concatenate(self*B2)
            return (self*target_matrix[0:m,0:p//2]).concatenate(self*target_matrix[0:m,p//2:p])
        else:
            # split A vertically and B horizontally
            #A1 = self[0:n,0:m//2]
            #A2 = self[0:n,m//2:m]
            #B1 = target_matrix[0:m//2,0:p]
            #B2 = target_matrix[m//2:m,0:p]
            #return A1*B1 + A2*B2
            return self[0:n,0:m//2]*target_matrix[0:m//2,0:p] + self[0:n,m//2:m]*target_matrix[m//2:m,0:p]


    
    # concatenate two matrices to form a new one
    def concatenate(self,other,dim='horizontal'):
        """Return a new matrix that is the concatenation of the calling matrix and another matrix.
        
        Arguments:
        other -- the matrix that is concatenated to the calling matrix
        dim -- string: the dimension along which the matrices are concatenated, either 'horizontal' where the width of the matrix increases or 'vertical' where the height of the matrix increases (default 'horizontal')
        """
        if dim == 'horizontal':
            new_data = [x + y for x,y in zip(self.data, other.data)]
        elif dim == 'vertical':
            new_data = self.data
            for i in other.data:
                new_data.append(i)
        return Matrix(new_data)
    
    # return scalar multiplied matrix
    def __scalar_multiplication(self,scalar):
        """Return the scalar product of the matrix with a scalar.
        
        Arguments:
        scalar -- a numeric value that scales all elements of the matrix
        """
        resulting_matrix = Matrix(self.n_rows,self.n_cols)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                resulting_matrix.set(i,j,scalar*self.data[i][j])
        return resulting_matrix
    
    # return determinant
    def get_determinant(self):
        """
        Return the determinant of a square matrix.
        Raises an exception if the matrix is not square.
        """
        if not self.is_square():
            raise Exception("Cannot calculate determinant because matrix is not square! Matrix is " +  str(self.n_rows) + "x" + str(self.n_cols))
            return 0
        det = 0
        
        # create a deep copy of the calling matrix to avoid modifying it when calculating row echelon form
        calling_matrix = deepcopy(self)

        # transform the matrix to a normal row echelon form
        calling_matrix.__transform_to_row_echelon_form()
                    
        det = calling_matrix.get_diagonal_product()
        return det

    # check if matrix is singular
    def is_singular(self) -> bool:
        """Return true if the determinant of the matrix is zero. Otherwise, return false."""
        return self.get_determinant() == 0
    
    # return trace
    def get_trace(self):
        """
        Return the trace of a square matrix.
        Raises an exception if the matrix is not square.
        """
        if not self.is_square():
            raise Exception("Cannot calculate trace because matrix is not square! Matrix is " +  str(self.n_rows) + "x" + str(self.n_cols))
            return 0
        return self.get_diagonal_sum()
        
    # return product of diagonal elements
    def get_diagonal_product(self):
        """Return the product of all the diagonal elements in the matrix."""
        product = self.data[0][0]
        for i in range(1,self.n_cols):
            product = product*self.data[i][i]
        return product
    
    # return sum of diagonal elements
    def get_diagonal_sum(self):
        """Return the sum of all the diagonal elements in the matrix."""
        sum = 0
        for i in range(self.n_rows):
            sum = sum + self.data[i][i]
        return sum
    
    # return diagonal elements in a list
    def get_diagonal_elements(self):
        """Return the diagonal elements of a matrix as a list."""
        diagonal = []
        for i in range(self.n_rows):
            diagonal.append(self.data[i][i])
        return diagonal

    # check if matrix is invertible
    def is_invertible(self) -> bool:
        """Return true if the matrix is not singular. Otherwise, return false."""
        return not self.is_singular()
    
    # make the matrix an identity matrix
    def make_identity(self) -> None:
        """Set all diagonal elements of the matrix to 1 and all non-diagonal elements to 0."""
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if i == j:
                    self.set(i,j,1)
                else:
                    self.set(i,j,0)

    def append_column(self, column) -> None:
        """Append a column to the right of the matrix.
        
        Arguments:
        column --- list or single-column matrix: the column that is appended to the right of the matrix
        
        Raises an exception if 'column' is not a list or a matrix.
        """
        if isinstance(column,list):
            for i in range(self.n_rows):
                self.data[i].append(column[i])
        elif isinstance(column,Matrix):
            for i in range(self.n_rows):
                self.data[i].append(column.data[i][0])
        else:
            raise Exception("Could not detect type of column to append!")
        self.n_cols = self.n_cols+1
    
    def append_row(self, row) -> None:
        """Append a row to the bottom of the matrix.
        
        Arguments:
        row --- list or single-row matrix: the row that is appended to the bottom of the matrix
        
        Raises an exception if 'row' is not a list or a matrix.
        """
        if isinstance(row,list):
            self.data.append(row)
        elif isinstance(row,Matrix):
            self.data.append(row.get_row[0])
        else:
            raise Exception("Could not detect type of row to append!")
        self.n_rows = self.n_rows+1

    # check if matrix is symmetric
    def is_symmetric(self) -> bool:
        """Return true if the matrix equals its transpose. Otherwise, return false."""
        return self == self.get_transpose()


    # transform the parameter matrix to row echelon form; is another matrix is also passed, use it as the augmented matrix
    def __transform_to_row_echelon_form(self, augmented_matrix=None, calculate_rank=False):
        """
        Modify the matrix so that it is transformed to a row echelon form using Gauss-Jordan elimination.
        This row echelon form is not the reduced row echelon form.
        
        Arguments:
        augmented_matrix -- optional matrix (usually identity) that is subjected to the same row operations as the calling matrix (default None)
        calculate_rank -- Boolean: whether to return the rank of the calling matrix (default false)
                
        The augmented matrix is used in calculating the inverse of a matrix.
        """

        # loop through columns
        for j in range(0,self.n_cols):
            
            # this check ensures that the algorithm works also for non-square matrices (e.g., when we want to calculate their rank instead of inverting them)
            if self.n_rows > j:
                first_row = self.get_row(j)
            else:
                continue
            
            # loop through rows
            for i in range(1+j,self.n_rows):
                # zero the element in the first column using type 3 row operations (add to one row the scalar multiple of another)
            
                # get the row we are trying to modify
                current_row = self.get_row(i)

                # if the current element is already 0, continue
                if current_row[0+j] == 0:
                    continue
                
                # check if the row is zero and if it is, continue to next one (no need to operate on a row of zeros)
                if self.row_is_zeros(j):
                    continue

                # calculate the scalar to multiply the first row with
                multiplier = current_row[0+j]/first_row[0+j]

                # perform type 3 row operations
                # first apply to the matrix we're currently working on
                self.__type_three_row_operation(current_row,first_row,multiplier)
                # then apply to augmented matrix
                if augmented_matrix is not None:
                    self.__type_three_row_operation(augmented_matrix.get_row(i),augmented_matrix.get_row(j),multiplier)
        
        # finally, count how many zero rows we have to be able to calculate rank
        if calculate_rank:
            n_zero_rows = self.count_zero_rows()
            return self.n_rows-n_zero_rows


    def get_row_echelon_form(self):
        return deepcopy(self).__transform_to_row_echelon_form()

    def count_zero_rows(self) -> int:
        """Return the number of rows that have only zero-valued elements."""
        num_zero_rows = 0
        for row in range(self.n_rows):
            if self.row_is_zeros(row):
                num_zero_rows = num_zero_rows + 1
        return num_zero_rows

    def row_is_zeros(self, r: int) -> bool:
        """
        Return true if all elements in the row are zero-valued. Otherwise, return false.
        
        Arguments:
        r -- unsigned integer: the index of the row
        """
        
        is_zero = True
        for elem in self.data[r]:
            if elem != 0:
                is_zero = False
                break
        return is_zero

    # return the rank of a matrix
    def get_rank(self):
        """
        Return the rank of a matrix.
        The rank is the number of linearly independent rows (or columns) in a matrix.
        """
        return deepcopy(self).__transform_to_row_echelon_form(calculate_rank=True)

    # return the inverse of a matrix
    def get_inverse(self):
        """
        Return the inverse matrix of a square matrix.
        The product of a matrix and its inverse matrix is an identity matrix.
        Raises an exception if the matrix is not square.
        """
        
        # create a deep copy of the calling matrix to avoid modifying it when calculating inverse
        calling_matrix = deepcopy(self)

        if not calling_matrix.is_square():
            raise Exception("Matrix is not invertible because it is not square! Matrix is " +  str(calling_matrix.n_rows) + "x" + str(calling_matrix.n_cols))
            return 0

        # create an augmented matrix that is initially an identity matrix
        augmented_matrix = Matrix(calling_matrix.n_rows,calling_matrix.n_cols,0)
        augmented_matrix.make_identity()

        # first, transform the matrix to a normal row echelon form
        calling_matrix.__transform_to_row_echelon_form(augmented_matrix)
                
        # if the determinant of the matrix is 0, it is singular and therefore not invertible
        if calling_matrix.get_diagonal_product() == 0:
            raise Exception ("Matrix is not invertible because it is singular!")

        # then, transform the row echelon form to reduced row echelon form
        # in the first part, set the leading coefficients to 1 with type 2 row operations (multiply a row by a scalar)
        for i in range(0,calling_matrix.n_rows):
            multiplier = 0
            current_row = calling_matrix.get_row(i)
            for c in range(calling_matrix.n_cols):
                if current_row[c] == 0:
                    continue
                elif current_row[c] != 0 and multiplier == 0:
                    multiplier = 1./current_row[c]
                    break

            if multiplier != 0:
                calling_matrix.__type_two_row_operation(current_row,multiplier)
                calling_matrix.__type_two_row_operation(augmented_matrix.get_row(i),multiplier)
            
        # in the second part, the elements on each row to the right of the leading coefficient to zero with type 3 row operations
        for i in range(calling_matrix.n_rows-1,-1,-1):
            reference_row = calling_matrix.get_row(i)
            for j in range(i-1,-1,-1):
                operable_row = calling_matrix.get_row(j)
                leading_found = False
                multiplier = 0
                for c in range(0,calling_matrix.n_cols):
                    # check if is leading coefficient
                    if operable_row[c] != 0 and not leading_found:
                        leading_found = True
                        continue
                    if leading_found and operable_row[c] != 0 and reference_row[c] != 0:
                        multiplier = operable_row[c]/reference_row[c]

                # if we have a reason to perform type 3 operations, we do so
                if leading_found and multiplier != 0:
                    calling_matrix.__type_three_row_operation(operable_row,reference_row,multiplier)
                    calling_matrix.__type_three_row_operation(augmented_matrix.get_row(j),augmented_matrix.get_row(i),multiplier)
                
        # return the final inverted matrix
        return augmented_matrix
                        
    # perform type 3 row operation (add the scalar multiple of multiplied_row to operable_row)
    def __type_three_row_operation(self,operable_row,multiplied_row,scalar):
        """Perform a type three row operation (add the scalar multiple of a row to another row)."""
        for c in range(self.n_cols):
            operable_row[c] = operable_row[c] - multiplied_row[c]*scalar
            
    # perform type 2 row operation (multiply operable row by a scalar)
    def __type_two_row_operation(self,operable_row,scalar):
        """Perform a type two row operation (multiply a row by a scalar)."""
        for c in range(self.n_cols):
            operable_row[c] = operable_row[c]*scalar

    # REPLACED BY [] OPERATOR
    #def get_rows_columns(self,rows_list,cols_list):
    #    """Return a submatrix from specified indices."""
    #    col = []
    #    for i in rows_list:
    #        row = []
    #        for j in cols_list:
    #            row.append(self.data[i][j])
    #        col.append(row)
    #    return Matrix(col)
    
    # REPLACED BY [] OPERATOR
    #def set_rows_columns(self,rows_list,cols_list,matrix):
    #    """Set the values in specified indices."""
    #    rows = matrix.get_height()
    #    cols = matrix.get_width()
    #    rows_first = rows_list[0]
    #    cols_first = cols_list[0]
    #    for i in rows_list:
    #        for j in cols_list:
    #            self.data[i][j] = matrix[i-rows_first][j-cols_first]
            
    def norm_Euclidean(self):
        """Return the Euclidean norm of the matrix."""
        norm = 0
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                norm = norm + pow(self.data[i][j],2)
        return math.sqrt(norm)

    # doesn't seem to work
    def get_dominant_eigenvector(self):
        eigenvector = []
        for i in range(self.n_cols):
            eigenvector.append(random.randint(-10,10))

        eigenvector = Matrix(eigenvector)

        err = 1e9

        while err > 1e-15:
            eigenvector_prev = eigenvector
            eigenvector = self*eigenvector
            eigenvector = eigenvector*(1/eigenvector.norm_Euclidean())
            err = (eigenvector-eigenvector_prev).norm_Euclidean()
            
        return eigenvector 
            


class LabeledMatrix(Matrix):
    """
    A child class of Matrix that has labeled columns.
    
    Attributes:
    data -- nested list: contains the numeric values in the matrix, implemented as a list of lists that represent the rows of the matrix
    n_rows -- unsigned integer: the number of rows in the matrix, also known as its height
    n_cols -- unsigned integer: the number of columns in the matrix, also known as its width
    labels_cols -- list of strings: the unique labels of the columns of the matrix
    labels_rows -- list of strings: the unique labels of the rows of the matrix
    """
    
    labels_rows = []
    labels_cols = []

    def __init__(self, *args, labels=None, column=True, labels_col = None, labels_row = None):
        
        # ensure that either labels is given or labels_col and labels_row are given, but both labels and labels_col/labels_row cannot be given
        if labels != None and (labels_col != None or labels_row != None):
            raise Exception("You cannot define both labels and labels_col or labels_row.")
        
        # if labels is given, use it to initialize the labels of the matrix (either row labels or columns labels depending on the value of 'column') 
        if labels != None and column == True:
            self.labels_cols = labels
        elif labels != None and column == False:
            self.labels_rows = labels
            
        # if labels_col or labels_row are given, use them to initialize the labels of the matrix
        if labels_col != None:
            self.labels_cols = labels_col
        if labels_row != None:
            self.labels_rows = labels_row
            
        # ensure that all label elements are unique (we don't want duplicate labels because we must be able to use them unambiguously to access specific rows and columns)
        if self.labels_cols != None:
            if len(self.labels_cols) != len(set(self.labels_cols)):
                raise Exception("Column labels must be unique!")
            if len(self.labels_rows) != len(set(self.labels_rows)):
                raise Exception("Row labels must be unique!")

        # if the number of non-label-related arguments is 1, we either have a dictionary argument or arguments according to the parent class
        if len(args) == 1:
            # check to see if the argument is a dictionary and if so, unpack that dictionary to generate the data and the labels
            if isinstance(args[0], dict):
                data = [x for x in args[0].values()]
                # if the labels are for columns, transpose the data so that it is in the correct format
                if column:
                    data = list(map(list, zip(*data)))
                    self.labels_cols = list(args[0].keys())
                else:
                    self.labels_rows = list(args[0].keys())
                super().__init__(data)
            # if no dictionary is present, we can refer to the parent's implementation of the constructor
            elif isinstance(args[0], list):
                super().__init__(args[0])
            else:
                raise Exception("LabeledMatrix must be initialized with a dictionary or a list of lists!")
        else:
            super().__init__(*args)
            

    # override __str__
    def __str__(self, precision = 4, space = 20):
        """
        Return a string that describes the matrix when string() is called with the matrix as the argument.
        Row and column labels are shown in left and top borders, respectively.
        This function defines the string representation of LabeledMatrix.
        """
        
        # first, populate an f-string with column labels
        if len(self.labels_rows) > 0:
            fstring_labels = f"{'':^{space}}"
        else:
            fstring_labels = f"{'':^0}"
        for label in self.labels_cols:
            fstring_labels = fstring_labels + f"{label:^{space}}"

        # fstring_total will contain the whole matrix (data and labels) as a single f-string  
        fstring_total = fstring_labels
        # populate each row of the f-string starting with the row label and then the values on that row
        for i in range(self.n_rows):
            fstring_row = f"{'':^0}"
            if len(self.labels_rows) > 0:
                fstring_row = fstring_row + f"{self.labels_rows[i]:^{space}}"
            for j in range(self.n_cols):
                matrix_string = str(round(self.data[i][j],precision))
                fstring_row = fstring_row + f"{matrix_string:^{space}}"
            # at the end of each row, we should add a line break
            fstring_total = fstring_total + '\n' + fstring_row
        return fstring_total

    # override print
    def print(self, precision = 4, space = 20):
        """
        Print a string that describes the labeled matrix.
        
        Calls the overloaded str() function with the possibility to specify how many decimals are shown.
        
        Arguments:
        precision -- unsigned integer: the number of decimals shown in the output (default 4)
        """
        print(self.__str__(precision, space))
        
    def get_column(self, c: str, as_list=True):
        """
        Return a specified column of the matrix as list or matrix.
        
        Arguments:
        c -- str: label of the column
        as_list - Boolean: whether to return the column as a list or not, in which case it is returned as a matrix (default true)
        """
        if isinstance(c,str):
            idx = self.labels_cols.index(c)
            return super().get_column(idx, as_list)
        else:
            return super().get_column(c, as_list)
    

    def get_row(self, r: str, as_list=True):
        """
        Return a specified row of the matrix as list or matrix.
        
        Arguments:
        r -- str: label of the row
        as_list - Boolean: whether to return the row as a list or not, in which case it is returned as a matrix (default true)
        """
        if isinstance(r,str):
            idx = self.labels_rows.index(r)
            return super().get_row(idx, as_list)
        else:
            return super().get_row(r, as_list)


    def __getitem__(self,label):
        """
        Return a matrix from the specified labels.
        Overloads the get[] operator.
        
        Arguments:
        label -- str, or a two-element tuple of strings
        
        Calls the parent's __getitem__() if label is not of expected type.
        """
        # first, check if the given label is a string (only one row is queried)
        if isinstance(label,str):
            # if the label is a string, return a matrix with the specified row
            return super().__getitem__(self.labels_rows.index(label))
        # otherwise, check if the given index is a tuple of strings (i.e., a value from a given row and column is queried)
        elif isinstance(label,tuple):
            row,col = label
            # rows is str and cols is str (one value from given row and column indices is queried)
            if isinstance(row,str) and isinstance(col,str):
                return super().__getitem__((self.labels_rows.index(row),self.labels_cols.index(col)))
            else:
                return super().__getitem__(label)

            
        else:
            return super().__getitem__(label)

    # then to set data
    def __setitem__(self,label,value):
        """
        Set elements of the matrix.
        Overloads the set[] operator.
        
        Arguments:
        label -- str, or a two-element tuple of strings
        
        Calls the parent's __setitem__() if label is not of expected type.
        """
        if isinstance(label,tuple):
            row, col = label
            super().__setitem__((self.labels_rows.index(row),self.labels_cols.index(col)), value)
        elif isinstance(label,str):
            super().__setitem__(self.labels_rows.index(label), value)
            
        else:
            return super().__setitem__(label,value)
        
    # set a row at given label to given values from a list
    def set_row(self, r: str, new_row: list):
        """
        Set the specified row of the matrix.

        Arguments:
        r -- str: label of the row
        new_row -- list: the values that are assigned to that row
        """
        super().set_row(self.labels_rows.index(r), new_row)

    # set a single value in given labels
    def set(self, row: str, col: str, value):
        """
        Set the element at specified labels.
        
        Arguments:
        row -- str: label of the row
        col -- str: label of the column
        value -- numeric value: the value that is assigned to the specified element
        """
        super().set(self.labels_rows.index(row), self.labels_cols.index(col), value)
        
    # get a single value in given labels
    def get(self, row: str, col: str):
        """
        Get the element at specified position.
        
        Arguments:
        row -- str: label of the row
        col -- str: label of the column
        """
        return super().get(self.labels_rows.index(row), self.labels_cols.index(col))