from random import random
from copy import deepcopy
import math
from mola.matrix import Matrix
from mola.utils import zeros, get_mean, uniques, randoms

# calculate the Euclidean distance between two points; note that this actually returns the squared distance, but because we only need it to compare distances, it doesn't matter and not including the square root is faster to compute
def distance_euclidean_pow(p1,p2):
    """Return the squared Euclidean distance between two points.
    If you want to retrieve the actual Euclidean distance, take the square root of the result. However, using this squared version is computationally more efficient.
    
    Arguments:
    p1 -- list: the first point
    p2 -- list: the second point
    """
    
    if isinstance(p1,Matrix):
        p1 = p1.get_column(0)
    if isinstance(p2,Matrix):
        p2 = p2.get_column(0)
    distance = 0
    for i in range(len(p1)):
        distance = distance + pow(p1[i]-p2[i],2)
    return distance


def distance_taxicab(p1,p2):
    """
    Return the taxicab distance (also known as Manhattan distance) between two points.
    
    Arguments:
    p1 -- list: the first point
    p2 -- list: the second point
    """
    distance = 0
    for i in range(len(p1)):
        distance = distance + abs(p1[i]-p2[i])
    return distance


# hard k-means clustering algorithm
def find_k_means(data, num_centers = 2, max_iterations = 100, distance_function = distance_euclidean_pow, initial_centers = None):
    """
    Return the cluster centers using hard k-means clustering.
    
    Note that there is no guarantee that the algorithm converges. This is why you should use several restarts or fuzzy k-means (function find_c_means() in this module).
    
    Arguments:
    data -- Matrix: the data containing the points to be clustered
    num_centers -- int: the number of centers to be found (default 2)
    max_iterations -- int: the maximum number of iterations where cluster centers are updated (default 100)
    distance_function -- function: the distance function to be used (default Euclidean distance); options are squared Euclidean distance (distance_euclidean_pow) and taxicab distance (distance_taxicab)
    initial_centers -- Matrix: the initial cluster centers; if not specified, they are initialized randomly (default None)
    """


    # get the dimension of the data
    dim = data.get_width()
    num_points = data.get_height()
    closest_center = [0 for x in range(num_points)]

    # if no initial centers are given, initialize centers with random floating-point values between 0 and 1
    if initial_centers is None:
        centers = randoms(num_centers,dim)
        # if the centers are not unique, generate new ones until they are
        while len(uniques(centers)) < num_centers:
            centers = randoms(num_centers,dim)
    else:
        centers = initial_centers
    previous_centers = deepcopy(centers)
    

    for iteration in range(max_iterations):

        # assignment step: assign each point to closest cluster center
        for row in range(num_points):
            distance = math.inf
            current_center = 0
            for i in range(num_centers):
                distance_to_center = distance_function(centers[i,:],data[row,:])
                if distance_to_center < distance:
                    distance = distance_to_center
                    closest_center[row] = i
        
        

        # update step: update the position of centers to be the mean of the points assigned to them
        for i in range(num_centers):
            points_in_cluster = []
            for row in range(num_points):
                if closest_center[row] == i:
                    points_in_cluster.append(data.get_row(row))
            if len(points_in_cluster) > 0:
                centers[i,:] = get_mean(points_in_cluster).get_transpose()
            
        # if the centers remained the same as in previous iteration, break out of the loop
        if centers == previous_centers:
            print("k-means centers converged at iteration ", str(iteration+1))
            break
        
        # set previous centers to current centers
        previous_centers = deepcopy(centers)
        
        # if we reach the maximum number of iterations, warn that it wasn't enough
        if iteration == max_iterations-1:
            print("WARNING: k-means centers did not converge in " , str(max_iterations), " iterations. Consider increasing the maximum number of iterations or using fuzzy k-means.")

    return centers




# soft k-means clustering algorithm
def find_c_means(data, num_centers = 2, max_iterations = 100, distance_function = distance_euclidean_pow, initial_centers = None):
    """
    Return the cluster centers and the membership matrix of points using soft k-means clustering (also known as fuzzy c-means).
    
    This algorithm is well-suited to cluster data that is not clearly separable into distinct clusters.
        
    Arguments:
    data -- Matrix: the data containing the points to be clustered
    num_centers -- int: the number of cluster centers to be found (default 2)
    max_iterations -- int: the maximum number of iterations where cluster centers are updated (default 100)
    distance_function -- function: the distance function to be used (default Euclidean distance); options are squared Euclidean distance (distance_euclidean_pow) and taxicab distance (distance_taxicab)
    initial_centers -- Matrix: the initial cluster centers; if not specified, they are initialized randomly (default None)
    """

    def update_membership_matrix():
        """
        Update the membership matrix U.
        The function loops through each point in the data and calculates the membership value for each cluster center into the membership matrix.
        """
        nonlocal U
        for row in range(num_points):
            for c in range(num_centers):
                distance_to_center = distance_function(centers[c],data.get_row(row))
                U[row,c] = 1 / sum([pow(distance_to_center/distance_function(centers[j],data.get_row(row)),2/(m-1)) for j in range(num_centers)])
        
    def update_centers():
        """
        Update the cluster centers.
        """
        nonlocal centers
        for c in range(num_centers):
            numerator = zeros(1,dim)
            denominator = 0
            for row in range(num_points):
                numerator = numerator + pow(U.get(row,c),m)*(data[row,:].get_transpose())
                denominator = denominator + pow(U[row,c],m)
            centers[c,:] = numerator / denominator

    # threshold to stop iterating
    threshold = 1e-9

    # get the dimension of the data
    dim = data.get_width()
    num_points = data.get_height()
    
    # if user has not defined initial centers, initialize centers with random floating-point values between 0 and 1; each row in the matrix is a cluster center
    if initial_centers is None:
        centers = randoms(num_centers,dim)
        # if the centers are not unique, generate new ones until they are
        while len(uniques(centers)) < num_centers:
            #centers = Matrix([[random() for x in range(dim)] for y in range(num_centers)])
            centers = randoms(num_centers,dim)
    else:
        centers = initial_centers
    
    # initialize weighing constant m
    m = 2.0
    
    # initialize the membership matrix U; it has as many rows as there are points and as many columns as there are centers; therefore, the value U[i,j] describes how strongly point i belongs to cluster center j
    U = zeros(num_points,num_centers)
    update_membership_matrix()
    previous_U = deepcopy(U)

    for iteration in range(max_iterations):

        update_centers()
        update_membership_matrix()
        
        # if the membership matrix U remained the same as in previous iteration, break out of the loop
        if abs(U.norm_Euclidean() - previous_U.norm_Euclidean()) < threshold:
            print("fuzzy k-means centers converged at iteration ", str(iteration+1))
            break
        
        # set previous centers to current centers
        previous_U = deepcopy(U)
        
        # if we reach the maximum number of iterations, warn that it wasn't enough
        if iteration == max_iterations-1:
            print("WARNING: fuzzy k-means centers did not converge in " , str(max_iterations), " iterations. Consider increasing the maximum number of iterations.")

    return centers, U

