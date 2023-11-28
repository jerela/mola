from asyncio.windows_events import INFINITE
from json.encoder import INFINITY
from mola.matrix import Matrix
from mola.utils import zeros, get_mean, uniques, randoms, norm
from random import random
from copy import deepcopy
import math

# calculate the Euclidean distance between two points; note that this actually returns the squared distance, but because we only need it to compare distances, it doesn't matter and not including the square root is faster to compute
def distance_euclidean_pow(p1,p2) -> float:
    """
    Return the squared Euclidean distance between two points.
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


def distance_euclidean(p1,p2) -> float:
    """
    Return the Euclidean distance between two points.
    
    Arguments:
    p1 -- list: the first point
    p2 -- list: the second point
    """
    return math.sqrt(distance_euclidean_pow(p1,p2))


def distance_taxicab(p1,p2) -> float:
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
def find_k_means(data: Matrix, num_centers = 2, max_iterations = 100, distance_function = distance_euclidean_pow, initial_centers = None):
    """
    Return the cluster centers using hard k-means clustering.
    
    K-means clustering is an iterative algorithm that finds the cluster centers by first assigning each point to the closest cluster center and then updating the cluster centers to be the mean of the points assigned to them. This process is repeated for a set number of iterations or until the cluster centers converge. The initial cluster centers are either randomized or given by the user.
    
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
def find_c_means(data: Matrix, num_centers = 2, max_iterations = 100, distance_function = distance_euclidean_pow, initial_centers = None):
    """
    Return the cluster centers and the membership matrix of points using soft k-means clustering (also known as fuzzy c-means).
    
    Fuzzy c-means clustering is an iterative algorithm that finds the cluster centers by first assigning each point to each cluster center with a certain membership value (0 to 1) and then updating the cluster centers to be the weighted mean of the points assigned to them. This process is repeated for a set number of iterations or until the cluster centers converge. The initial cluster centers are either randomized or given by the user.
    A major difference between hard k-means clustering and fuzzy c-means clustering is that in fuzzy c-means clustering, the points may belong partially to several clusters instead of belonging completely to one cluster, like in hard k-means clustering. Therefore, this algorithm is well-suited to cluster data that is not clearly separable into distinct clusters (e.g., symmetric distribution of data points).
        
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


# density-based subtractive clustering
def find_density_clusters(data: Matrix, num_centers = 2, beta = 0.5, sigma = 0.5):
    """
    Return the cluster centers using density-based subtractive clustering.
    
    Density-based subtractive clustering is an iterative algorithm that finds the cluster centers by first calculating the mountain function for each point and then selecting the peaks of the mountain functions as cluster centers. The mountain function is calculated by summing the Gaussian functions centered at each point.
    
    Arguments:
    data -- Matrix: the data containing the points to be clustered
    num_centers -- int: the number of cluster centers to be found (default 2)
    beta -- float: the width of the Gaussian function (default 0.5)
    sigma -- float: the width of the Gaussian function (default 0.5)
    """

    # get the number of data points (samples) and the dimension of each data point
    n_samples = data.get_height()
    dim = data.get_width()

    # initialize the values of mountain functions to zero
    #mountain_func = zeros(n_samples, 1)
    mountain_func = [0 for x in range(n_samples)]

    # construct mountain function value for each data sample
    # iterate through centers
    for i in range(n_samples):
        # iterate through data points
        for k in range(n_samples):
            mountain_func[i] = mountain_func[i] + math.exp( - ( pow(distance_euclidean(data[i,:],data[k,:]),2) ) / (2*sigma**sigma) )

    # select cluster centers and destruct mountain functions
    mountain_func_new = deepcopy(mountain_func)

    c_subtractive = zeros(num_centers,dim)

    # iterate through the number of labels (assumption is that there are 2 clusters)
    for k in range(num_centers):
        # select cluster centers
        peak = 0;
        peak_i = 0;
        for i in range(n_samples):
            if mountain_func_new[i] > peak:
                print('For cluster ' + str(k) + ' found peak ' + str(mountain_func_new[i]) + ' at ' + str(data[i,0]) + ',' + str(data[i,1]))
                peak = mountain_func_new[i]
                peak_i = i;
            
    
        

        # save cluster centers
        c_subtractive[k,:] = data[peak_i,:].get_transpose()
    
        # destruct mountain functions
        for i in range(n_samples):
            mountain_func_new[i] = mountain_func_new[i] - mountain_func_new[peak_i] * math.exp( - ( pow(distance_euclidean(data[i,:],c_subtractive[k,:]),2)) / (2*beta**beta) )

    # assign all data points to a cluster depending on the distance
    labeled_subtractive = [0 for x in range(n_samples)]
    for i in range(n_samples):
        cluster = 0
        prev_norm = 1e6
        for k in range(num_centers):
            if norm(data[i,:]-c_subtractive[k,:]) < prev_norm:
                prev_norm = norm(data[i,:]-c_subtractive[k,:])
                cluster = k;
        labeled_subtractive[i] = cluster
    
    return labeled_subtractive

