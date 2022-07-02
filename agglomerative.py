import numpy as np

class Agglomerative_Clustering:
    def __init__(self, k, init_k):
        self.k = k
        self.init_k = init_k
        self.d = int(256 / (self.init_k))

    def merge_clusters(self, points):
        self.clusters_list = self.init_clusters(points)
        # iterate through all clusters until get 5 clustera
        while len(self.clusters_list) > self.k:
            # get the closest pair of clusters by using euclidean distance
            cluster1, cluster2 = min([(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]], key=lambda c:self.clusters_distance(c[0], c[1]))
            self.clusters_list = [c for c in self.clusters_list if c != cluster1 and c != cluster2]
            merged_cluster = cluster1 + cluster2
            self.clusters_list.append(merged_cluster)
        self.cluster = {}
        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num
        self.centers = {}
        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)

    def predict_center(self, point):
        # Find center of each of 5 clusters
        center = self.centers[self.cluster[tuple(point)]]
        return center

    def euclidean_distance(self,point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def clusters_distance(self, cluster1, cluster2):
        cluster1_center = np.average(cluster1, axis=0)
        cluster2_center = np.average(cluster2, axis=0)
        return self.euclidean_distance(cluster1_center, cluster2_center)

    def init_clusters(self, points):
        # assign points to different clusters to generate collection of clusters which equal number of points
        classes = {}
        for k in range(self.init_k):
            j = k * self.d
            classes[(j, j, j)] = []
        for l, p in enumerate(points):
            go = min(classes.keys(), key=lambda c: self.euclidean_distance(p, c))
            classes[go].append(p)
        return [g for g in classes.values() if len(g) > 0]



