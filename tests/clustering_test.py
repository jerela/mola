from mola import Matrix
from mola import clustering
from mola import utils


def test_k_means_clustering():
    initial_centers = Matrix([[0,0],[20,0]])
    symmetric_points = Matrix([[-1,0],[-2,2],[1,1],[20,2],[18,0],[22,-1],[23,-1]])
    centers = clustering.find_k_means(data=symmetric_points,num_centers=2,initial_centers=initial_centers)
    assert(utils.equals_approx(centers,Matrix([[-0.6667, 1.0],[20.75, 0.0]]),precision = 1e-4))

def test_c_means_clustering():
    initial_centers = Matrix([[-1,0],[1,0]])
    symmetric_points = Matrix([[-1,1],[-2,2],[1,1],[2,2],[0,0],[-1,-1],[1,-1]])
    centers, membership = clustering.find_c_means(data=symmetric_points,num_centers=2,initial_centers = initial_centers)
    assert(utils.equals_approx(centers,Matrix([[-1.2182, -0.7022],[1.2182, 0.7022]]),precision = 1e-4))
    