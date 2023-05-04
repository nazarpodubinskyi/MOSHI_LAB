import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output


N = 1000

X, y = make_blobs(n_samples=N, centers=None, n_features=2, random_state=0, center_box=(1, 40), cluster_std=1.5)
print(X)

data = pd.DataFrame(X, columns=['x', 'y'])

data.to_csv('out.csv', index=False)


class K_Klustering:
    def __init__(self, data, k):
        self.data = data
        self.k = k



class Main:
    def __init__(self, data, option):
        if (option == 1):
            K_Klustering()


# 1. Scale the data (min max scaling)
def scale_data(data, multiply_by=1, add_num=0):
    scaled_data = ((data - data.min()) / (data.max() - data.min())) * multiply_by + add_num
    return scaled_data


# 2. Initialize random centroids
def random_centroids(data, k):
    centroids = []
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)


def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)


# 4. Update centroids
def new_centroids(data, labels, k):
    return data.groupby(labels).apply(lambda x: np.exp(np.log(x)).mean()).T


# visualizing process
def plot_clusters(data, labels, centroids, iteration):
    plt.clf()
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)

    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=labels)
    plt.scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1])
    plt.draw()
    plt.pause(0.2)


plt.ion()
# scaled_data=scale_data(data=data)
scaled_data = data
max_iterations = 1000
k = 4

centroids = random_centroids(data=scaled_data, k=k)
old_centroids = pd.DataFrame()

iteration = 1

while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids

    labels = get_labels(data=scaled_data, centroids=centroids)

    centroids = new_centroids(data=data, labels=labels, k=k)
    plot_clusters(data=scaled_data, labels=labels, iteration=iteration, centroids=centroids)
    iteration += 1

#plt.waitforbuttonpress()
