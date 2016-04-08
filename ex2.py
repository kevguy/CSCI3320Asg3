from __future__ import print_function

import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import misc
from struct import unpack
from time import time

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def plot_mean_image(X, log):
    meanrow = X.mean(0)
    # present the row vector as an image
    plt.figure(figsize=(3,3))
    plt.imshow(np.reshape(meanrow,(28,28)), cmap=plt.cm.binary)
    plt.title('Mean image of ' + log)
    plt.show()

def get_labeled_data(imagefile, labelfile):
    """
    Read input-vector (image) and target class (label, 0-9) and return it as list of tuples.
    Adapted from: https://martin-thoma.com/classify-mnist-with-pybrain/
    """
    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Read the binary data
    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    X = np.zeros((N, rows * cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        for id in range(rows * cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            X[i][id] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (X, y)


def my_clustering(X, y, n_clusters, pca):
    # =======================================
    # Complete the code here.
    # return scores like this: return [score, score, score, score]
    # =======================================
    from sklearn.cluster import KMeans
    #print('fuck X ', X.shape)
    #print('fuck y ', y.shape)
    clf = KMeans(n_clusters)
    clf.fit(X)

    from sklearn import metrics
    ari = metrics.adjusted_rand_score(y, clf.labels_)
    mri = metrics.adjusted_mutual_info_score(y, clf.labels_)
    v_measure = metrics.v_measure_score(y, clf.labels_)
    '''
    silhouette_coeff = metrics.silhouette_score(X, clf.labels_,
                                      metric='euclidean',
                                      sample_size=300)
    '''
    silhouette_coeff = metrics.silhouette_score(X, clf.labels_)

    show_images(n_clusters, clf, pca)


    return [ari,mri,v_measure,silhouette_coeff]


def POV_arr(eigenvalues):
    arr = []
    sum = 0
    for i in range(0, len(eigenvalues)):
        sum += eigenvalues[i]

    cumulate = 0
    for i in range(0, len(eigenvalues)):
        cumulate += eigenvalues[i]
        arr.append(cumulate / sum)
    return arr


def show_images(n_clusters, kmean, pca):

    # in kmean.cluster_centers_.shape, what we get is a (n_clusters, 84) array
    # so for every [1,84] vector we have inside kmean.cluster_centers_,
    # we need to trasform it back to the original vector
    # then transorm it back into the original 28x28 matrix   
    cluster_centers_org = pca.inverse_transform(kmean.cluster_centers_)


    n_row = 1
    n_col = n_clusters
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(cluster_centers_org[i].reshape(28,28), cmap=plt.cm.gray)
        title_text = 'Center ' + str(i + 1)
        plt.title(title_text, size=12)
        plt.xticks(())
        plt.yticks(())

    plt.savefig('ex2_n_clusters_' + str(n_clusters) + '.png')
    # plt.show()



def main():
    # Load the dataset
    fname_img = 't10k-images.idx3-ubyte'
    fname_lbl = 't10k-labels.idx1-ubyte'
    [X, y]=get_labeled_data(fname_img, fname_lbl)

    # Dataset info
    print('Total dataset size: ')
    # each row of matrix X represents an image
    print("n_samples: ", X.shape[0])
    print("n_features: ", X.shape[1])

    # Plot the mean image
    plot_mean_image(X, 'all images')

    # =======================================
    # Complete the code here.
    # Use PCA to reduce the dimension here.
    # You may want to use the following codes. Feel free to write in your own way.
    # - pca = PCA(n_components=...)
    # - pca.fit(X)
    # - print('We need', pca.n_components_, 'dimensions to preserve 0.95 POV')
    # =======================================

    pca = PCA(n_components=0.95).fit(X)
    X_pca = pca.transform(X)

    # Clustering
    range_n_clusters = [8, 9, 10, 11, 12]
    ari_score = [None] * len(range_n_clusters)
    mri_score = [None] * len(range_n_clusters)
    v_measure_score = [None] * len(range_n_clusters)
    silhouette_avg = [None] * len(range_n_clusters)


    print(79 * '_')
    print('% 9s' % 'n_clusters'
      '     time               ARI               MRI               v-measure score    avg silhouette score')


    for n_clusters in range_n_clusters:
        t0 = time()
        i = n_clusters - range_n_clusters[0]
        #print("Number of clusters is: ", n_clusters)
        [ari_score[i], mri_score[i], v_measure_score[i], silhouette_avg[i]] = my_clustering(X_pca, y, n_clusters, pca)
        # print('The ARI score is: ', ari_score[i])
        # print('The MRI score is: ', mri_score[i])
        # print('The v-measure score is: ', v_measure_score[i])
        # print('The average silhouette score is: ', silhouette_avg[i])
        print('% 9s   %.15f   %.15f   %.15f   %.15f    %.15f'
          % (n_clusters, (time() - t0), ari_score[i], mri_score[i], v_measure_score[i], silhouette_avg[i]))

    print(79 * '_')

    # =======================================
    # Complete the code here.
    # Plot scores of all four evaluation metrics as functions of n_clusters.
    # =======================================
    # Plot Adjusted Rand Index
    plt.figure()
    plt.plot(range_n_clusters, ari_score, color='b', linestyle='-',label='ARI')
    plt.title('Adjusted Rand Index')
    plt.ylabel('score')
    plt.xlabel('number of clusters')
    #plt.legend(loc='upper left', prop={'size':6})
    plt.savefig('ex2_score_ARI.png')

    # Plot Mutual Information based scores
    plt.figure()
    plt.plot(range_n_clusters, mri_score, color='r', linestyle='-',label='MRI')
    plt.title('Mutual Information based scores')
    plt.ylabel('score')
    plt.xlabel('number of clusters')
    #plt.legend(loc='upper left', prop={'size':6})
    plt.savefig('ex2_score_MRI.png')

    # Plot V-measure
    plt.figure()
    plt.plot(range_n_clusters, v_measure_score, color='c', linestyle='-',label='V-measure')
    plt.title('V-measure')
    plt.ylabel('score')
    plt.xlabel('number of clusters')
    #plt.legend(loc='upper left', prop={'size':6})
    plt.savefig('ex2_score_V_measure.png')

    # Plot Silhouette Coefficient
    plt.figure()
    plt.plot(range_n_clusters, silhouette_avg, color='m', linestyle='-',label='Silhouette')
    plt.title('Silhouette Coefficient')
    plt.ylabel('score')
    plt.xlabel('number of clusters')
    #plt.legend(loc='upper left', prop={'size':6})
    plt.savefig('ex2_score_Silhouette.png')    


    plt.figure()
    plt.plot(range_n_clusters, ari_score, color='b', linestyle='-',label='ARI')
    plt.plot(range_n_clusters, mri_score, color='r', linestyle='-',label='MRI')
    plt.plot(range_n_clusters, v_measure_score, color='c', linestyle='-',label='V-measure')
    plt.plot(range_n_clusters, silhouette_avg, color='m', linestyle='-',label='Silhouette')
    plt.title('Overall Scores')
    plt.ylabel('score')
    plt.xlabel('number of clusters')
    plt.legend(loc='upper left', prop={'size':6})
    plt.savefig('ex2_score_overall.png')
    #plt.show()

    # v-measure: range(0,1) 1 means perfectly complete labeling
    # ARI adjusted Rand index: is ensured to have a value close to 0.0
    #                            for random labeling independently of the 
    #                           number of clusters and samples and exactly 1.0 
    #                           when the clusterings are identical (up to a 
    #                           permutation).
    #                           Similarity score between -1.0 and 1.0. 
    #                           Random labelings have an ARI close to 0.0. 1.0 stands for perfect match.
    # The AMI: returns a value of 1 when the two partitions are identical (ie perfectly matched). 
    #            Random partitions (independent labellings) have an expected AMI around 0 on average hence can be negative.
    # Silhoutte: The best value is 1 and the worst value is -1. 
    #               Values near 0 indicate overlapping clusters. 
    #               Negative values generally indicate that a sample has been assigned 
    #               to the wrong cluster, as a different cluster is more similar.

if __name__ == '__main__':
    main()
