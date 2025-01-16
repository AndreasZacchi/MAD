import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


data = pd.read_csv('../housing.csv') # Load the data

unprocessed_data = data.to_numpy() # Convert the data to a numpy array

data = unprocessed_data[:,[0,6,7,8]] # Select MedInc, Latitude, Longitude, MedHouseVal columns

normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0) # Normalize the data
amount_of_clusters = 5

complete = AgglomerativeClustering(linkage="complete", n_clusters=amount_of_clusters).fit(normalized_data)
average = AgglomerativeClustering(linkage="average", n_clusters=amount_of_clusters).fit(normalized_data)
single = AgglomerativeClustering(linkage="single", n_clusters=amount_of_clusters).fit(normalized_data)

def calculate_intra_dist(centroids, data, labels, n_clusters, normalize=True):
    intra_cluster_distance = []
    for i in range(n_clusters): 
        points_in_cluster = data[labels == i] 
        centroid = centroids[i]
        
        dist = np.linalg.norm(points_in_cluster - centroid, axis=1)
        norm_dist = dist / len(points_in_cluster) # normalize the distance by number of points in cluster
        
        if normalize:
            to_append = norm_dist
        else:
            to_append = dist

        intra_cluster_distance.append(np.sum(to_append))
    return intra_cluster_distance

def calculate_inter_dist(centroids, n_clusters):
    inter_cluster_distance = []
    for i in range(n_clusters):
        for j in range(i+1, n_clusters): # i+1 to avoid calculating the same distance twice
            inter_cluster_distance.append(np.linalg.norm(centroids[i] - centroids[j]))
    return inter_cluster_distance



centroids_complete = np.array([normalized_data[complete.labels_ == i].mean(axis=0) for i in range(amount_of_clusters)])
centroids_average = np.array([normalized_data[average.labels_ == i].mean(axis=0) for i in range(amount_of_clusters)])
centroids_single = np.array([normalized_data[single.labels_ == i].mean(axis=0) for i in range(amount_of_clusters)])

norm_intra_dist_complete = calculate_intra_dist(centroids_complete, normalized_data, complete.labels_, amount_of_clusters)
norm_intra_dist_average = calculate_intra_dist(centroids_average, normalized_data, average.labels_, amount_of_clusters)
norm_intra_dist_single = calculate_intra_dist(centroids_single, normalized_data, single.labels_, amount_of_clusters)

intra_dist_complete = calculate_intra_dist(centroids_complete, normalized_data, complete.labels_, amount_of_clusters, normalize=False)
intra_dist_average = calculate_intra_dist(centroids_average, normalized_data, average.labels_, amount_of_clusters, normalize=False)
intra_dist_single = calculate_intra_dist(centroids_single, normalized_data, single.labels_, amount_of_clusters, normalize=False)

inter_dist_complete = calculate_inter_dist(centroids_complete, amount_of_clusters)
inter_dist_average = calculate_inter_dist(centroids_average, amount_of_clusters)
inter_dist_single = calculate_inter_dist(centroids_single, amount_of_clusters)

avg_cluster_size_complete = np.bincount(complete.labels_)
avg_cluster_size_average = np.bincount(average.labels_)
avg_cluster_size_single = np.bincount(single.labels_)

print("====================================================")
print("Complete linkage:")
print("====================================================")
print("Avg intra cluster distance:", np.mean(intra_dist_complete))
print("Avg normalised intra cluster distance:", np.mean(norm_intra_dist_complete))
print("Inter cluster distance:", np.mean(inter_dist_complete))
print("Cluster sizes:", avg_cluster_size_complete)
print("====================================================")
print("Average linkage:")
print("====================================================")
print("Avg intra cluster distance:", np.mean(intra_dist_average))
print("Avg normalised intra cluster distance:", np.mean(norm_intra_dist_average))
print("Inter cluster distance:", np.mean(inter_dist_average))
print("Cluster sizes:", avg_cluster_size_average)
print("====================================================")
print("Single linkage:")
print("====================================================")
print("Avg intra cluster distance:", np.mean(intra_dist_single))
print("Avg normalised intra cluster distance:", np.mean(norm_intra_dist_single))
print("Inter cluster distance:", np.mean(inter_dist_single))
print("Cluster sizes:", avg_cluster_size_single)

def plot_clusters(data, labels, centroids, title):
    plt.scatter(data[:,1], data[:,2], c=labels)
    plt.scatter(centroids[:,1], centroids[:,2], c='red', marker='x', s=100)
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.title(title)
    plt.savefig(title + ".png")

plot_clusters(normalized_data, complete.labels_, centroids_complete, "Complete linkage")
plot_clusters(normalized_data, average.labels_, centroids_average, "Average linkage")
plot_clusters(normalized_data, single.labels_, centroids_single, "Single linkage")

