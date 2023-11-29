from mola import Matrix
from mola import clustering
from mola import utils


def test_k_means_clustering():
    """Test hard k-means clustering."""
    initial_centers = Matrix([[0,0],[20,0]])
    symmetric_points = Matrix([[-1,0],[-2,2],[1,1],[20,2],[18,0],[22,-1],[23,-1]])
    centers = clustering.find_k_means(data=symmetric_points,num_centers=2,initial_centers=initial_centers)[0]
    assert(utils.equals_approx(centers,Matrix([[-0.6667, 1.0],[20.75, 0.0]]),precision = 1e-4))

def test_c_means_clustering():
    """Test fuzzy c-means clustering."""
    initial_centers = Matrix([[-1,0],[1,0]])
    symmetric_points = Matrix([[-1,1],[-2,2],[1,1],[2,2],[0,0],[-1,-1],[1,-1]])
    centers, membership = clustering.find_c_means(data=symmetric_points,num_centers=2,initial_centers = initial_centers)
    assert(utils.equals_approx(centers,Matrix([[-1.2182, -0.7022],[1.2182, 0.7022]]),precision = 1e-4))
    
def test_density_clustering():
    """Test density clustering."""
    density_points = Matrix([[1,2], [0.5,1.5], [0,1], [0,0.5], [0,0], [0,-0.5], [0,-1], [0.5,-1.5], [1,-2], [2,0], [2.5,-0.5], [3,-1], [3,-1.5], [3,-2], [3,-2.5], [3,-3], [2.5,-3.5], [2,-4], [0.3, -0.7], [0.3, 0.7], [-0.3, -0.7], [-0.3, 0.7], [2.7, -2.7], [3.3, -2.7], [2.7, -1.3], [3.3, -1.3]])
    centers_density, labels = clustering.find_density_clusters(data=density_points, num_centers=2, beta=0.5, sigma=0.5)
    assert(labels == [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])