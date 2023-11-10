from mola import Matrix, LabeledMatrix
from mola import regression
from mola import utils
from mola import decomposition


def test_equals_approx():
    """Test the equals_approx() method of utils, which we will be using in comparing some matrices."""
    assert(utils.equals_approx([1,2],[1,2]))
    assert(utils.equals_approx(3.5,3.5))
    assert(utils.equals_approx((1,2),(1,2)))
    assert(utils.equals_approx(Matrix(1,2),Matrix(1,2)))

def test_inverse():
    """Test matrix inversion."""
    matrix = Matrix([[2,1,-1],[-3,-1,2],[-2,1,2]])
    inverse = matrix.get_inverse()
    identity_matrix = utils.identity(3)
    product = matrix*inverse
    assert(product==identity_matrix)
        
def test_read_write():
    """Test reading and writing matrices to files."""
    mat1 = Matrix([ [1,2,3],[4,5,6],[5,6,7] ])
    utils.write_matrix_to_file(mat1,'data.txt')
    mat2 = utils.read_matrix_from_file('data.txt')
    assert(mat2==Matrix([ [1,2,3],[4,5,6],[5,6,7] ]))
        
def test_multiplication():
    """Test matrix multiplication."""
    mat1 = Matrix(3,5,1)
    mat2 = utils.identity(3,3)
    assert(mat2*mat1==mat1)
    
def test_singularity():
    """Test matrix singularity and invertibility functions."""
    # create a singular matrix
    sing = Matrix([[4, 2, 0], [2, 1, 0], [5,12,-5]])
    assert(sing.is_invertible() == False)
    assert(sing.is_singular() == True)
    
def test_rank():
    """Test the get_rank() method of Matrix."""
    sing = Matrix([[4, 2, 0], [2, 1, 0], [5,12,-5]])
    A = Matrix([[1, 1, 0, 2], [-1, -1, 0, -2]])
    B = Matrix([[5]])
    assert(sing.get_rank() == 2)
    assert(A.get_rank() == 1)
    assert(B.get_rank() == 1)


# TESTS FOR LABELED MATRICES BEGIN HERE
def test_labeled_from_dict():
    """Test the constructor of LabeledMatrix that takes a dictionary as an argument."""
    labeled_matrix_dict = {'first_column': [1, 2, 3], 'its_square': [1, 4, 9]}
    labeled_matrix = LabeledMatrix(labeled_matrix_dict)
    matrix = Matrix([[1, 1], [2, 4], [3, 9]])
    assert(labeled_matrix == matrix)

def test_labeled_from_lists():
    """Test the constructor of LabeledMatrix that takes a list of lists as an argument."""
    labeled_matrix = LabeledMatrix([[1,2,3], [4,5,6], [7,8,9]], labels_col = ['first column', 'second column', 'third column'], labels_row=['first row', 'second row', 'third row'])
    matrix = Matrix([[1,2,3], [4,5,6], [7,8,9]])
    assert(labeled_matrix == matrix)

def test_labeled_get_set():
    """Test the get() and set() methods of LabeledMatrix."""
    labeled_matrix = LabeledMatrix([[1,2,3], [4,5,6], [7,8,9]], labels_col = ['first column', 'second column', 'third column'], labels_row=['first row', 'second row', 'third row'])
    labeled_matrix.set('first row', 'third column', 0)
    assert(labeled_matrix.get('first row', 'third column') == 0)

def test_labeled_getitem():
    """Test the __getitem__() method of LabeledMatrix."""
    labeled_matrix = LabeledMatrix([[1,2,3], [4,5,6], [7,8,9]], labels_col = ['first column', 'second column', 'third column'], labels_row=['first row', 'second row', 'third row'])
    assert(labeled_matrix['first row'] == labeled_matrix[0])
    assert(labeled_matrix['second row','third column'] == labeled_matrix[1,2])
    
def test_labeled_setitem():
    """Test the __setitem__() method of LabeledMatrix."""
    labeled_matrix = LabeledMatrix([[1,2,3], [4,5,6], [7,8,9]], labels_col = ['first column', 'second column', 'third column'], labels_row=['first row', 'second row', 'third row'])   
    labeled_matrix['second row','third column'] = 0
    assert(labeled_matrix['second row','third column'] == 0)

def test_labeled_get_row():
    """Test the get_row() method of LabeledMatrix."""
    labeled_matrix = LabeledMatrix([[1,2,3], [4,5,6], [7,8,9]], labels_col = ['first column', 'second column', 'third column'], labels_row=['first row', 'second row', 'third row'])   
    assert(labeled_matrix.get_row('second row') == labeled_matrix.get_row(1))
    
def test_labeled_set_row():
    """Test the set_row() method of LabeledMatrix."""
    labeled_matrix = LabeledMatrix([[1,2,3], [4,5,6], [7,8,9]], labels_col = ['first column', 'second column', 'third column'], labels_row=['first row', 'second row', 'third row'])
    labeled_matrix.set_row('first row', [0,0,0])
    assert(labeled_matrix.get_row('first row') == [0,0,0])

def test_labeled_get_column():
    """Test the get_column() method of LabeledMatrix."""
    labeled_matrix = LabeledMatrix([[1,2,3], [4,5,6], [7,8,9]], labels_col = ['first column', 'second column', 'third column'], labels_row=['first row', 'second row', 'third row'])   
    assert(labeled_matrix.get_column('third column') == labeled_matrix.get_column(2))
    