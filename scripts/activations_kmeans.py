import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
import sys


class tSNE_KMeans(object):

    filename = ""
    paths = []
    points = []

    def parse(self, filename):
        self.filename = filename
        with open(filename) as file:
            data = json.load(file)
            for entry in data:
                path = entry['path']
                point = entry['point']
                self.paths.append(path)
                self.points.append(point)

    def cluster(self):
        print('Parsing ' + str(len(self.paths)) + ' data points')
        # data shaping
        pointarray = np.array(self.points)
        z = np.float32(pointarray)

        # kmeans
        num_clusters = 20
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(z, num_clusters, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        #plot
        for cluster in xrange(num_clusters):
            cluster_data = z[label.ravel() == cluster]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s = 2, c=np.random.rand(3,), lw = 0)
            plt.scatter(center[:, 0], center[:, 1], s=60, c='y', marker='s')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(self.filename + '.pdf')

    def clear(self):
        self.paths = []
        self.points = []
        plt.clf()
        plt.close()


if __name__ == '__main__':
    filename = sys.argv [1]
    kmeans = tSNE_KMeans()

    kmeans.parse(filename)
    kmeans.cluster()
    kmeans.clear()
    kmeans.clear()