import unittest
from molalib import Matrix

class MatrixTestCase(unittest.TestCase):

    def setUp(self):
        self.matrix = Matrix(3,3,1)

    def test_A(self):
        self.fail("Not implemented")

    def test_inverse(self):
        self.matrix.inverse()

if __name__ == '__main__':
    unittest.main()
