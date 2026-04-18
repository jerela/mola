from mola import Matrix
from mola import regression
from mola import utils
from mola import decomposition
import time

start = time.time()


mat1 = utils.read_matrix_from_file('data/mat10x10.txt')
mat2 = mat1

#mat1 = utils.randoms(50,50)
#mat2 = utils.randoms(50,50)
print("Starting loop")
for i in range(10000):
    #mat1 = Matrix([[1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9]])
    #mat2 = utils.identity(9,9)



    #mat1 = Matrix([[1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9]])
    #mat2 = utils.identity(9,18)
    #mat1 = Matrix([[1,2, 3],[4,5,6],[7,8,9]])
    #mat2 = utils.identity(3,3)
    #mat1 = Matrix([[1,2],[3,4],[5,6]])
    #mat2 = utils.identity(2,2)
    mat3 = mat1*mat2

end = time.time()
print("Loop done")
print(end - start)

#mat3.print()
