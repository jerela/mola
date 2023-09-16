import unittest
from mola import Matrix
from mola import regression
from mola import utils

class MatrixTestCase(unittest.TestCase):

    def setUp(self):
        self.matrix = Matrix(3,3,1)

    def test_inverse(self):
        matrix = Matrix([[2,1,-1],[-3,-1,2],[-2,1,2]])
        inverse = matrix.get_inverse()
        identity_matrix = utils.identity(3)
        product = matrix*inverse
        self.assertEqual(product,identity_matrix)
        
    def test_read_file(self):
        mat = utils.read_matrix_from_file('data.txt')
        self.assertEqual(mat,Matrix([ [1,2,3],[4,5,6],[5,6,7] ]))
        
    def test_multiplication(self):
        mat2 = Matrix(3,5,1)
        mat3 = mat2.transpose()
        mat4 = mat2.matrix_multiplication(mat3)
        mat5 = mat2*mat3
        self.assertEqual(mat4,mat5)
        
    def test_regression(self):
        H = Matrix([[2,1],[4,1],[6,1]])
        y = Matrix([[0],[1],[2]])
        th = regression.linear_least_squares(H,y)
        self.assertAlmostEqual(th,(0.5,-1))


if __name__ == '__main__':
    unittest.main()
