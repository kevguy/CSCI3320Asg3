from __future__ import print_function
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics

def create_data():
    # Generate sample points
    centers = [[3,5], [5,1], [8,2], [6,8], [9,7]]
    X, y = make_blobs(n_samples=1000,centers=centers,cluster_std=0.5,random_state=3320)
    # =======================================
    # Complete the code here.
    # Plot the data points in a scatter plot.
    # Use color to represents the clusters.
    # =======================================

    # We end up with X and y
    # X:    array of shape [n_samples, n_features]
    # y:    array of shape [n_samples]
    # Let's choose five colors for our labels
    # b: blue, g: green, r: red, c: cyan, m: magenta
    y_color = []
    for i in range(0, y.shape[0]):
        if (y[i] == 0):
            y_color.append('b')
        elif (y[i] == 1):
            y_color.append('g')
        elif (y[i] == 2):
            y_color.append('r')
        elif (y[i] == 3):
            y_color.append('c')
        elif (y[i] == 4):
            y_color.append('m')   
            
    plt.figure()
    # plot the data points in empty circles
    for i in range(0, X.shape[0]):
        plt.scatter(X[i][0], X[i][1], facecolors='none', edgecolor = y_color[i])
    # plot the centres with full color and edge with black
    plt.scatter(3,5, color = 'b', edgecolor = 'k', marker = 'x', s = 256, linewidths=3)
    plt.scatter(5,1, color = 'g', edgecolor = 'k', marker = 'x', s = 256, linewidths=3)
    plt.scatter(8,2, color = 'r', edgecolor = 'k', marker = 'x', s = 256, linewidths=3)
    plt.scatter(6,8, color = 'c', edgecolor = 'k', marker = 'x', s = 256, linewidths=3)
    plt.scatter(9,7, color = 'm', edgecolor = 'k', marker = 'x', s = 256, linewidths=3)

    plt.savefig('original_data.png')
    plt.show()


    return [X, y]

def my_clustering(X, y, n_clusters):
    # =======================================
    # Complete the code here.
    # return scores like this: return [score, score, score, score]
    # =======================================
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters = n_clusters)
    clf.fit(X)
    pred = clf.predict(X)

    y_color = []
    for i in range(0, pred.shape[0]):
        if (pred[i] == 0):
            y_color.append('b')
        elif (pred[i] == 1):
            y_color.append('g')
        elif (pred[i] == 2):
            y_color.append('r')
        elif (pred[i] == 3):
            y_color.append('c')
        elif (pred[i] == 4):
            y_color.append('m')
        elif (pred[i] == 5):
            y_color.append('y')
            
    plt.figure()
    # plot the data points in empty circles
    for i in range(0, X.shape[0]):
        plt.scatter(X[i][0], X[i][1], facecolors='none', edgecolor = y_color[i])

    # plot the centres with full color and edge with black
    for i in range(0, clf.cluster_centers_.shape[0]):
        plt.scatter(clf.cluster_centers_[i][0], clf.cluster_centers_[i][1], marker = 'x', color = 'k', s = 256, linewidths=3)        

    plt.title('number of clusters: ' + str(n_clusters))
    plt.savefig('ex1_n_clusters_' + str(n_clusters) + '.png')
    plt.show()

    return [0,0,0,0]  # You won't need this line when you are done

def main():
    X, y = create_data()
    range_n_clusters = [2, 3, 4, 5, 6]
    ari_score = [None] * len(range_n_clusters)
    mri_score = [None] * len(range_n_clusters)
    v_measure_score = [None] * len(range_n_clusters)
    silhouette_avg = [None] * len(range_n_clusters)

    for n_clusters in range_n_clusters:
        i = n_clusters - range_n_clusters[0]
        print("Number of clusters is: ", n_clusters)
        [ari_score[i], mri_score[i], v_measure_score[i], silhouette_avg[i]] = my_clustering(X, y, n_clusters)
        print('The ARI score is: ', ari_score[i])
        print('The MRI score is: ', mri_score[i])
        print('The v-measure score is: ', v_measure_score[i])
        print('The average silhouette score is: ', silhouette_avg[i])

    # =======================================
    # Complete the code here.
    # Plot scores of all four evaluation metrics as functions of n_clusters.
    # =======================================

if __name__ == '__main__':
    main()

