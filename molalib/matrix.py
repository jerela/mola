from copy import deepcopy
from re import I
from unittest import defaultTestLoader
from xml.dom.expatbuilder import makeBuilder

class Matrix:
    """
    A class that represents a mathematical matrix used in linear algebra tasks. The number of rows and columns are defined by you and settable. Methods include transpose, inverse, norms, etc.
    :param data: contains the numeric values in the matrix, implemented as a list of lists that represent the rows of the matrix
    :param n_rows: unsigned int: the number of rows in the matrix, also known as its height
    :param n_cols: unsigned int: the number of columns in the matrix, also known as its width
    """
    

    n_rows = 0
    n_cols = 0
    data = list

    # construct a matrix with r rows, c columns, and some initial value (default 0)
    def __init__(self,r,c,value=0):
        self.n_rows = r
        self.n_cols = c
        #self.data = [[value]*c]*r
        col = []
        for j in range(r):
            row = []
            for i in range(c):
                row.append(value)
            col.append(row)
        self.data = col
            
    # overload equals (==) operator
    def __eq__(self, other):
        # first check that dimensions match; if not, return false
        if self.n_rows != other.n_rows | self.n_cols != other.n_cols:
            raise Exception("Matrix dimensions do not match.")
            return 0
        
        # assume that the matrices are equal; compare each element and if any exists that isn't equal, change the assumption to false
        equals = True

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.get(i,j) != other.get(i,j):
                    equals = False
        return equals

    # overload multiplication (*) operator
    def __mul__(self, other):
        if isinstance(self,Matrix) and isinstance(other,Matrix):
            return self.matrix_multiplication(other)
        elif isinstance(self,Matrix) and isinstance(other,int):
            return self.scalar_multiplication(Matrix,other)
        elif isinstance(self,Matrix) and isinstance(other,float):
            return self.scalar_multiplication(Matrix,other)
        else:
            raise Exception("Cannot identify type of term on right when multiplying!")

    # return the number of rows
    def getRows(self):
        return self.n_rows
    
    # return the number of columns
    def getCols(self):
        return self.n_cols
    
    # return a row as a list
    def getRow(self,r):
        return self.data[r]
    
    # set a row at given index to given values from a list
    def setRow(self,r,new_row):
        self.data[r] = new_row

    # set a single value in a given index
    def set(self,i,j,value):
        self.data[i][j] = value

    # get a single value in a given index
    def get(self,i,j):
        return self.data[i][j]

    # print matrix in MATLAB-style format
    def print(self):
        matrix_string = '['
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                matrix_string = matrix_string + str(self.data[i][j])
                if j < self.n_cols-1:
                    matrix_string = matrix_string + ", "
            if i < self.n_rows-1:
                matrix_string = matrix_string + "; "
        matrix_string = matrix_string + "]"
        print(matrix_string)

    # transpose a matrix
    def transpose(self):
        transposed = Matrix(self.n_cols,self.n_rows)
        for i in range(self.n_cols):
            for j in range(self.n_rows):
                transposed.set(i,j,self.data[j][i])
        return transposed
    
    # return the transpose of a matrix
    def get_transpose(self):
        calling_matrix = deepcopy(self)
        calling_matrix.transpose()
        return calling_matrix

    # return matrix product
    def matrix_multiplication(self,target_matrix):
        n_rows = self.n_rows
        n_cols = target_matrix.getCols()
        product_matrix = Matrix(n_rows,n_cols)
        for i in range(n_rows):
            for j in range(n_cols):
                new_elem = 0
                length = self.n_cols
                for x in range(length):
                    new_elem = new_elem + self.data[i][x]*target_matrix.get(x,j)
                product_matrix.set(i,j,new_elem)
        return product_matrix
    
    # return scalar multiplied matrix
    def scalar_multiplication(self,scalar):
        resulting_matrix = Matrix(self.n_rows,self.n_cols)
        for i in range(n_rows):
            for j in range(n_cols):
                resulting_matrix.set(i,j,scalar*self.get(i,j))
        return resulting_matrix
    
    # return determinant
    def get_determinant(self):
        if not self.is_square():
            raise Exception("Cannot calculate determinant because matrix is not square! Matrix is " +  str(calling_matrix.n_rows) + "x" + str(calling_matrix.n_cols))
            return 0
        det = 0
        
        # create a deep copy of the calling matrix to avoid modifying it when calculating row echelon form
        calling_matrix = deepcopy(self)

        # transform the matrix to a normal row echelon form
        calling_matrix.transform_to_row_echelon_form()
                    
        det = calling_matrix.get_diagonal_product()
        return det


    # check if matrix is singular
    def is_singular(self):
        return self.get_determinant() == 0
    
    # return trace
    def get_trace(self):
        if not self.is_square():
            raise Exception("Cannot calculate trace because matrix is not square! Matrix is " +  str(calling_matrix.n_rows) + "x" + str(calling_matrix.n_cols))
            return 0
        return self.get_diagonal_sum()
        
    # return product of diagonal elements
    def get_diagonal_product(self):
        product = self.get(0,0)
        for i in range(1,self.n_cols):
            product = product*self.get(i,i)
        return product
    
    # return sum of diagonal elements
    def get_diagonal_sum(self):
        sum = 0
        for i in range(self.n_rows):
            sum = sum + get(i,i)
        return sum

    # check if matrix is invertible
    def is_invertible(self):
        return not self.is_singular()
    
    # make the matrix an identity matrix
    def make_identity(self):
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if i == j:
                    self.set(i,j,1)
                else:
                    self.set(i,j,0)

    # check if matrix is symmetric
    def is_symmetric(self):
        return self == self.get_transpose()

    # check if matrix is square
    def is_square(self):
        return self.n_cols == self.n_rows

    # transform the parameter matrix to row echelon form; is another matrix is also passed, use it as the augmented matrix
    def transform_to_row_echelon_form(self, augmented_matrix=None):
        for j in range(0,self.n_cols):
            first_row = self.getRow(j)
            for i in range(1+j,self.n_rows):
                # zero the element in the first column using type 3 row operations (add to one row the scalar multiple of another)
            
                # get the row we are trying to modify
                current_row = self.getRow(i)
            
                # if the current element is already 0, continue
                if current_row[0+j] == 0:
                    continue
            
                # calculate the scalar to multiply the first row with
                multiplier = current_row[0+j]/first_row[0+j]
            
                # perform type 3 row operations
                # first apply to the matrix we're currently working on
                self.type_three_row_operation(current_row,first_row,multiplier)
                # then apply to augmented matrix
                if augmented_matrix is not None:
                    self.type_three_row_operation(augmented_matrix.getRow(i),augmented_matrix.getRow(j),multiplier)

    # return the inverse of a matrix
    def get_inverse(self):
        
        # TODO: separate functionality into functions for row echelon and reduced row echelon; leverage the properties of invertible matrices
        
        # create a deep copy of the calling matrix to avoid modifying it when calculating inverse
        calling_matrix = deepcopy(self)

        if not calling_matrix.is_square():
            raise Exception("Matrix is not invertible because it is not square! Matrix is " +  str(calling_matrix.n_rows) + "x" + str(calling_matrix.n_cols))
            return 0

        # create an augmented matrix that is initially an identity matrix
        augmented_matrix = Matrix(calling_matrix.n_rows,calling_matrix.n_cols,0)
        augmented_matrix.make_identity()

        # first, transform the matrix to a normal row echelon form
        calling_matrix.transform_to_row_echelon_form(augmented_matrix)
                
        # then, transform the row echelon form to reduced row echelon form
        # in the first part, set the leading coefficients to 1 with type 2 row operations (multiply a row by a scalar)
        for i in range(0,calling_matrix.n_rows):
            multiplier = 0
            current_row = calling_matrix.getRow(i)
            for c in range(calling_matrix.n_cols):
                if current_row[c] == 0:
                    continue
                elif current_row[c] != 0 and multiplier == 0:
                    multiplier = 1./current_row[c]
                    break

            if multiplier != 0:
                calling_matrix.type_two_row_operation(current_row,multiplier)
                calling_matrix.type_two_row_operation(augmented_matrix.getRow(i),multiplier)
            
        # in the second part, the elements on each row to the right of the leading coefficient to zero with type 3 row operations
        for i in range(calling_matrix.n_rows-1,-1,-1):
            reference_row = calling_matrix.getRow(i)
            for j in range(i-1,-1,-1):
                print(str(i) + ", " + str(j))
                operable_row = calling_matrix.getRow(j)
                leading_found = False
                multiplier = 0
                for c in range(0,calling_matrix.n_cols):
                    # check if is leading coefficient
                    if operable_row[c] != 0 and not leading_found:
                        leading_found = True
                        print("leading found at " + str(c))
                        continue
                    if leading_found and operable_row[c] != 0 and reference_row[c] != 0:
                        multiplier = operable_row[c]/reference_row[c]

                # if we have a reason to perform type 3 operations, we do so
                if leading_found and multiplier != 0:
                    calling_matrix.type_three_row_operation(operable_row,reference_row,multiplier)
                    calling_matrix.type_three_row_operation(augmented_matrix.getRow(j),augmented_matrix.getRow(i),multiplier)
                
        # return the final inverted matrix
        return augmented_matrix
                        
    # perform type 3 row operation (add the scalar multiple of multiplied_row to operable_row)
    def type_three_row_operation(self,operable_row,multiplied_row,scalar):
        for c in range(self.n_cols):
            operable_row[c] = operable_row[c] - multiplied_row[c]*scalar
            
    # perform type 2 row operation (multiply operable row by a scalar)
    def type_two_row_operation(self,operable_row,scalar):
        for c in range(self.n_cols):
            operable_row[c] = operable_row[c]*scalar

