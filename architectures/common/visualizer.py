import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from architectures.common.matrix_manipulation import compute_pca, compute_lda, normalize


def visualize_data(data):

    projected_data, lost_variance_information = compute_pca(data, 2, False)
    x = projected_data[:,0]
    y = projected_data[:,1]

    print("Lost variance information in PCA: {}".format(lost_variance_information))

    plt.scatter(x, y, c="blue", alpha=0.5)
    plt.show()


def visualize_clusters(data, clusters_labels, use_lda=True, file_path=None):

    data = normalize(data)
    projected_data = None
    lost_variance_information = None

    if use_lda:
        # TODO: this must be checked
        if len(np.unique(clusters_labels)) < 3:
            return
        projected_data, lost_variance_information = compute_lda(data, clusters_labels, 2)
    else:
        projected_data, lost_variance_information = compute_pca(data, 2, False)

    x = projected_data[:,0]
    y = projected_data[:,1]

    print("Lost variance information in dimensionality reduction for visualizing clusters: {}"
          .format(lost_variance_information))

    non_empty_labels = np.unique(clusters_labels)
    print("K={}".format(len(non_empty_labels)))

    color_map = plt.cm.get_cmap("hsv", len(non_empty_labels)+1)

    fig = plt.figure()

    for label in non_empty_labels:
        indices = np.where(clusters_labels == label)[0]
        x_values = np.take(x, indices, axis=0)
        y_values = np.take(y, indices, axis=0)
        plt.scatter(x_values, y_values, color=color_map(label), alpha=0.5, label=label)

    #plt.legend()
    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)


def visualize_clusters_distribution(clusters_labels, file_path=None):

    c = Counter(clusters_labels)

    most_common = c.most_common()
    if len(most_common) > 128:
        most_common = most_common[0:128]

    # Create a properly sized figure depending on the number of clusters
    fig = plt.figure(figsize=(max(len(most_common) * 0.3, 10), 10))

    for i, value in enumerate(most_common):
        plt.bar(i, value[1], width=.5, color='blue')

    # Force xticks to be equal to the labels (avoid decimals)
    plt.xticks(np.arange(len(most_common)), [element[0] for element in most_common], rotation='vertical')
    #plt.show(block=True)

    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)
