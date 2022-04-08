import os
import time
import json
import config
import matplotlib.pyplot as plt

from architectures.nscnet import NSCNet
from architectures.visualizer import visualize_clusters
from architectures.images_loader import import_image_np_dataset


def create_required_folders():
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)


def show_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show(block=True)


def create_inputs(dummy_dataset=False):
    start_time = time.time()
    print('inputs creation started...')
    inputs = import_image_np_dataset(config.IMAGES_PATH,
                                     (config.INPUT_SHAPE[0], config.INPUT_SHAPE[1]),
                                     config.RGB_NORMALIZATION)
    if dummy_dataset:
        print('using dummy dataset')
        inputs = inputs[:500]

    execution_time = time.time() - start_time
    print("inputs creation completed in {} seconds.".format(round(execution_time, 2)))

    return inputs


def nscnet(inputs):

    kmeans_cluster_args = {
        'config': {
            "n_clusters": 20
        },
        "n_clusters": 20,
        "method": "kmeans"
    }

    dbscan_cluster_args = {
        'config': {
            "eps": 2,
            "min_samples": 5,
            "metric": "euclidean"
        },
        "n_clusters": 20,
        "method": "dbscan"
    }

    results = {}

    for K in config.N_POSSIBLE_CLUSTERS:
        kmeans_cluster_args = {
            'config': {
                "n_clusters": K
            },
            "n_clusters": K,
            "method": "kmeans"
        }

        nscnet = NSCNet(config.INPUT_SHAPE, kmeans_cluster_args)

        nscnet.train_model(inputs)

        clustering_output, features = nscnet.compute_clusters(inputs)

        results[K] = clustering_output

        # visualize_clusters(features, clustering_output["labels"])

    with open(f'results/kmeans_{K}.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == '__main__':

    create_required_folders()

    inputs = create_inputs(dummy_dataset=True)
    nscnet(inputs)
