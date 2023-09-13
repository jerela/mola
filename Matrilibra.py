
from locale import currency
from tkinter import N
from unittest import defaultTestLoader


class Matrix:
    

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
            
    # overload multiplication operator
    def __mul__(self, other):
        if isinstance(self,Matrix) and isinstance(other,Matrix):
            return self.multiplyMatrix(other)

    # return the number of rows
    def getRows(self):
        return self.n_rows
    
    # return the number of columns
    def getCols(self):
        return self.n_cols
    
    # return a row as a list
    def getRow(self,r):
        return self.data[r]
    
    # set a row at given index to given values
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

    # return a transpose of a matrix
    def transpose(self):
        transposed = Matrix(self.n_cols,self.n_rows)
        for i in range(self.n_cols):
            for j in range(self.n_rows):
                transposed.set(i,j,self.data[j][i])
        return transposed

    # implement matrix multiplication
    def multiplyMatrix(self,target_matrix):
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
        # WIP
        return product_matrix
    
    # implement elementwise multiplication
    def multiply_elementwise(self,target_matrix):
        # WIP
        pass
    
    # calculate determinant
    def determinant():
        pass
    
    def isSingular(self):
        return self.determinant() == 0
    
    def isInvertible(self):
        return not self.isSingular()
    
    

    def inverse(self):
        
        # TODO: separate functionality into functions for row echelon and reduced row echelon; don't modify the calling matrix itself, but work on its copy instead; leverage the properties of invertible matrices
        
        if self.n_cols != self.n_rows:
            print("Matrix is not invertible because it is not square! Matrix is " +  str(self.n_rows) + "x" + str(self.n_cols))
            return 0

        # create an augmented matrix that is initially an identity matrix
        augmented_matrix = Matrix(self.n_rows,self.n_cols,0)
        for i in range(self.n_rows):
            augmented_matrix.set(i,i,1)
        print("identity:")
        augmented_matrix.print()
        



        # first, transform the matrix to a normal row echelon form
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
                self.type_three_row_operation(augmented_matrix.getRow(i),augmented_matrix.getRow(j),multiplier)
                
        # then, transform the row echelon form to reduced row echelon form
        # in the first part, set the leading coefficients to 1 with type 2 row operations (multiply a row by a scalar)
        for i in range(0,self.n_rows):
            multiplier = 0
            current_row = self.getRow(i)
            for c in range(self.n_cols):
                if current_row[c] == 0:
                    continue
                elif current_row[c] != 0 and multiplier == 0:
                    multiplier = 1./current_row[c]
                    break

            if multiplier != 0:
                self.type_two_row_operation(current_row,multiplier)
                self.type_two_row_operation(augmented_matrix.getRow(i),multiplier)
            
        # in the second part, the elements on each row to the right of the leading coefficient to zero with type 3 row operations
        for i in range(self.n_rows-1,-1,-1):
            current_row = self.getRow(i)
            for j in range(i-1,-1,-1):
                print(str(i) + ", " + str(j))
                operable_row = self.getRow(j)
                leading_found = False
                multiplier = 0
                for c in range(0,self.n_cols):
                    # check if is leading coefficient
                    if operable_row[c] != 0 and not leading_found:
                        leading_found = True
                        print("leading found at " + str(c))
                        continue
                    if leading_found and operable_row[c] != 0 and current_row[c] != 0:
                        multiplier = operable_row[c]/current_row[c]

                if leading_found and multiplier != 0:
                    self.type_three_row_operation(operable_row,current_row,multiplier)
                    self.type_three_row_operation(augmented_matrix.getRow(j),augmented_matrix.getRow(i),multiplier)
                
        
        return augmented_matrix
                        
    # perform type 3 row operation (add the scalar multiple of multiplied_row to operable_row)
    def type_three_row_operation(self,operable_row,multiplied_row,scalar):
        for c in range(self.n_cols):
            operable_row[c] = operable_row[c] - multiplied_row[c]*scalar
            
    # perform type 2 row operation (multiply operable row by a scalar)
    def type_two_row_operation(self,operable_row,scalar):
        for c in range(self.n_cols):
            operable_row[c] = operable_row[c]*scalar


mat = Matrix(3,5)
print(mat.get(1,0))
mat.print()
mat.set(0,1,3)
print(mat.get(0,1))
mat.print()

mat2 = Matrix(3,5,1)
mat2.print()

mat3 = mat2.transpose()


mat4 = mat2.multiplyMatrix(mat3)
mat4.print()

mat5 = mat2*mat3
mat5.print()

mat6 = Matrix(3,3,1)
mat6.set(0,0,2)
mat6.set(0,2,-1)
mat6.set(1,0,-3)
mat6.set(1,1,-1)
mat6.set(1,2,2)
mat6.set(2,0,-2)
mat6.set(2,2,2)
mat6.print()
augmented_matrix = mat6.inverse()
print("row echeloned:")
mat6.print()

print("augmented:")
augmented_matrix.print()


mat7 = Matrix(3,3,1)
mat7.set(0,0,2)
mat7.set(0,2,-1)
mat7.set(1,0,-3)
mat7.set(1,1,-1)
mat7.set(1,2,2)
mat7.set(2,0,-2)
mat7.set(2,2,2)
mat7.print()
mat8 = mat7*augmented_matrix
mat8.print()



