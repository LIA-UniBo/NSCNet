import matplotlib.pyplot as plt
import numpy as np

from architectures.matrix_manipulation import compute_pca, compute_lda, normalize


def visualize_data(data):

    projected_data, lost_variance_information = compute_pca(data, 2, False)
    x = projected_data[:,0]
    y = projected_data[:,1]

    print("Lost variance information in PCA: {}".format(lost_variance_information))

    plt.scatter(x, y, c="blue", alpha=0.5)
    plt.show()


def visualize_clusters(data, clusters_labels, use_lda=True):

    data = normalize(data)
    projected_data = None
    lost_variance_information = None

    if use_lda:
        projected_data, lost_variance_information = compute_lda(data, clusters_labels, 2)
    else:
        projected_data, lost_variance_information = compute_pca(data, 2, False)

    x = projected_data[:,0]
    y = projected_data[:,1]

    print("Lost variance information in dimensionality reduction: {}".format(lost_variance_information))

    non_empty_labels = np.unique(clusters_labels)

    color_map = plt.cm.get_cmap("hsv", len(non_empty_labels)+1)

    for label in non_empty_labels:
        indices = np.where(clusters_labels == label)[0]
        x_values = np.take(x, indices, axis=0)
        y_values = np.take(y, indices, axis=0)
        plt.scatter(x_values, y_values, color=color_map(label), alpha=0.5, label=label)

    #plt.legend()
    plt.show()
