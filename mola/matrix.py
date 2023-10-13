from copy import deepcopy
import math
import random


class Matrix:
    """
    Class that represents a real matrix used in linear algebra tasks. The number of rows and columns are defined by you and settable. Methods include transpose, inverse, norms, etc.
    
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
        Return a matrix of specified dimensions.
        
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
        # first check if lists is just an int
        if isinstance(lists,int):
            self.n_rows = 1
            self.n_cols = 1
            col = []
            col.append([lists])
            self.data = col
        # check if list is more than 1D (assumedly 2D)
        elif isinstance(lists[0],list):
            self.n_rows = len(lists)
            self.n_cols = len(lists[0])
            col = []
            for j in range(self.n_rows):
                row = lists[j]
                col.append(row)
            self.data = col
        elif isinstance(lists,list):
            self.n_rows = len(lists)
            self.n_cols = 1
            col = []
            for j in range(self.n_rows):
                row = lists[j]
                col.append([row])
            self.data = col        

    def __abs__(self):
        if self.n_rows == 1 and self.n_cols == 1:
            return abs(self.data[0][0])
        
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
        if isinstance(idx, slice) or isinstance(idx,int):
            return Matrix(self.data[idx])
        elif isinstance(idx,tuple):
            rows,cols = idx
            # rows is int and cols is slice:
            if isinstance(rows,int) and isinstance(cols,slice):
                sliced_data = Matrix(self.data[rows][cols])
            # rows is int and cols is int
            elif isinstance(rows,int) and isinstance(cols,int):
                sliced_data = self.data[rows][cols]
            # rows is slice and cols is int
            elif isinstance(rows,slice) and isinstance(cols,int):
                sliced_data = Matrix([r[cols] for r in self.data[rows]])
            return sliced_data

            #if isinstance(rows,int):
            #    sliced_data = self.data[rows][cols]
            #else:
            #    sliced_data = [r[cols] for r in self.data[rows]]
            #return Matrix(sliced_data)
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
        # first check that dimensions match; if not, return false
        if self.n_rows != other.n_rows | self.n_cols != other.n_cols:
            raise Exception("Matrix dimensions do not match.")
            return 0
        
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
        
        Arguments:
        other -- Matrix or scalar, the term that is multiplied with the matrix
        
        If 'other' is a Matrix, the output is a matrix that is the product of the two matrices.
        If 'other' is an int or a float, the output a matrix whose elements have been multiplied by 'other'.
        
        Raises an exception if 'other' is not Matrix, int or float.
        """
        if isinstance(self,Matrix) and isinstance(other,Matrix):
            return self.__matrix_multiplication(other)
        elif isinstance(self,Matrix) and isinstance(other,int):
            return self.__scalar_multiplication(other)
        elif isinstance(self,Matrix) and isinstance(other,float):
            return self.__scalar_multiplication(other)
        else:
            print(type(other))
            raise Exception("Cannot identify type of term on right when multiplying!")
    
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
        if isinstance(other,Matrix):
            if other.n_rows == 1 and other.n_cols == 1:
                return self.data[0][0]/other
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
        if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
            raise Exception("Matrix dimensions must match for elementwise addition or subtraction!")
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                output.set(i,j,self[i][j]+other[i][j])
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
        if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
            raise Exception("Matrix dimensions must match for elementwise addition or subtraction!")
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
    def get_row(self,r,as_list=True):
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
    
    def get_column(self,c,as_list=True):
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
    def set_row(self,r,new_row):
        """Set the specified row of the matrix.

        Arguments:
        r -- unsigned integer: index of the row
        new_row -- list: the values that are assigned to that row
        """
        self.data[r] = new_row

    # set a single value in a given index
    def set(self,i,j,value):
        """Set the element at specified position."""
        self.data[i][j] = value

    # get a single value in a given index
    def get(self,i,j):
        """Get the element at specified position."""
        return self.data[i][j]

    # print matrix in MATLAB-style format
    def print(self, precision = 4):
        """
        Print a string that describes the matrix.
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
        print(matrix_string)

    # check if matrix elements are real
    def is_real(self):
        """Return true if all elements of the matrix are real-valued. Otherwise, return false."""
        real = True
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if not isinstance(self.data[i][j],float) and not isinstance(self.data[i][j],int):
                    real = False
        return real

    def is_identity(self):
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

    def is_square(self):
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

    # return matrix product
    def __matrix_multiplication(self,target_matrix):
        """Return the matrix product of two matrices.
        
        Arguments:
        target_matrix --- the matrix on the right side of multiplication
        """
        n_rows = self.n_rows
        n_cols = target_matrix.get_width()
        product_matrix = Matrix(n_rows,n_cols)
        for i in range(n_rows):
            for j in range(n_cols):
                new_elem = 0
                length = self.n_cols
                for x in range(length):
                    new_elem = new_elem + self.data[i][x]*target_matrix.data[x][j]
                product_matrix.set(i,j,new_elem)
        return product_matrix
    
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
    def is_singular(self):
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
    def is_invertible(self):
        """Return true if the matrix is not singular. Otherwise, return false."""
        return not self.is_singular()
    
    # make the matrix an identity matrix
    def make_identity(self):
        """Set all diagonal elements of the matrix to 1 and all non-diagonal elements to 0."""
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if i == j:
                    self.set(i,j,1)
                else:
                    self.set(i,j,0)

    def append_column(self,column):
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
    
    def append_row(self,row):
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
    def is_symmetric(self):
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

    def count_zero_rows(self):
        """Return the number of rows that have only zero-valued elements."""
        num_zero_rows = 0
        for row in range(self.n_rows):
            if self.row_is_zeros(row):
                num_zero_rows = num_zero_rows + 1
        return num_zero_rows

    def row_is_zeros(self,r):
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
            
