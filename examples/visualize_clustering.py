from mola import Matrix, LabeledMatrix
from mola import clustering
import matplotlib.pyplot as plt

# This script uses matplotlib to visualize three clustering algorithms (hard k-means, fuzzy c-means, density-based mountain clustering) with 2D points

# first, let's define some data points
data_points = Matrix([[1,2], [0.5,1.5], [0,1], [0,0.5], [0,0], [0,-0.5], [0,-1], [0.5,-1.5], [1,-2], [2,0], [2.5,-0.5], [3,-1], [3,-1.5], [3,-2], [3,-2.5], [3,-3], [2.5,-3.5], [2,-4], [0.3, -0.7], [0.3, 0.7], [-0.3, -0.7], [-0.3, 0.7], [2.7, -2.7], [3.3, -2.7], [2.7, -1.3], [3.3, -1.3]])
# conduct various clusterings
centers_density, labels_density = clustering.find_density_clusters(data=data_points, num_centers=2, beta = 0.5, sigma = 0.5)
centers_kmeans, labels_kmeans = clustering.find_k_means(data=data_points, num_centers = 2)
centers_cmeans, membership = clustering.find_c_means(data=data_points,num_centers=2)

# for plotting, we plot cluster centers with black x's and data point with circles; different clusters are in different colors, although in fuzzy c-means the color slides by the degree of membership in a cluster
fig, axs = plt.subplots(1,3)

fig.text(x=0.05,y=0.2,s="note how the standard hard k-means cannot correctly cluster the boundary points, but density clustering can")

# plot k-means clustering
i = 0
for point in data_points.data:
    if labels_kmeans[i] == 0:
        axs[0].plot(point[0],point[1],'bo')
    else:
        axs[0].plot(point[0],point[1],'ro')
    i += 1

for center in centers_kmeans.data:
    axs[0].plot(center[0],center[1],'kx')

axs[0].set_xlim(-4, 5)
axs[0].set_ylim(-5, 4)
axs[0].set_aspect('equal', adjustable='box')
axs[0].set_title('hard k-means')

# plot density clustering
i = 0
for point in data_points.data:
    if labels_density[i] == 0:
        axs[1].plot(point[0],point[1],'bo')
    else:
        axs[1].plot(point[0],point[1],'ro')
    i += 1
    
for center in centers_density.data:
    axs[1].plot(center[0],center[1],'kx')
axs[1].set_xlim(-4, 5)
axs[1].set_ylim(-5, 4)
axs[1].set_aspect('equal', adjustable='box')
axs[1].set_title('density clustering')

# plot fuzzy c-means clustering
i = 0
for point in data_points.data:
    axs[2].plot(point[0],point[1],'o', color=(membership[i,0],0,membership[i,1]))
    i += 1

for center in centers_cmeans.data:
    axs[2].plot(center[0],center[1],'kx')
axs[2].set_xlim(-4, 5)
axs[2].set_ylim(-5, 4)
axs[2].set_aspect('equal', adjustable='box')
axs[2].set_title('fuzzy c-means')

plt.show()

