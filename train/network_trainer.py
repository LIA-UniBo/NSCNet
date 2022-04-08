import abc
import json
import config
import itertools

import numpy as np

from architectures.nscnet import NSCNet
from architectures.visualizer import visualize_clusters


class NetworkTrainer:

    def __init__(self, network_name, result_dir):
        self.network_name = network_name
        self.results_dir = result_dir

    def dbscan(self, inputs):
        method = 'dbscan'
        config_list = list(itertools.product(config.EPS_VALUES, config.MIN_POINTS, config.N_POSSIBLE_CLUSTERS))
        for eps, min_points, K in config_list:
            cluster_args = {
                'config': {
                    "eps": eps,
                    "min_samples": min_points,
                    "metric": "euclidean",
                    "n_clusters": K
                },
                "n_clusters": K,
                "method": method
            }

            self.train(K, cluster_args, inputs, method)

    def kmeans(self, inputs):
        method = 'kmeans'

        for K in config.N_POSSIBLE_CLUSTERS:
            print('*' * 40)
            print(f'CLUSTERING METHOD: {method}')
            print(f'TRAINING THE NETWORK WITH {K} CLUSTERS')
            print('*' * 40)

            cluster_args = {
                'config': {
                    "n_clusters": K
                },
                "n_clusters": K,
                "method": method
            }

            self.train(K, cluster_args, inputs, method)

        print("\n\n")

    @abc.abstractmethod
    def train(self, **kwargs):
        return

    def save_training_results(self, K, clustering_output, features, clustering_method):

        # Adapt the content of the clustering_output so it can be saved as a json
        for elem in clustering_output:
            if isinstance(clustering_output[elem], np.float32):
                clustering_output[elem] = float(clustering_output[elem])
            elif isinstance(clustering_output[elem], np.ndarray):
                clustering_output[elem] = clustering_output[elem].tolist()

        file_name = f'{self.results_dir}/{self.network_name}_{clustering_method}_{K}'

        # Save image showing clusters
        visualize_clusters(features, clustering_output["labels"], file_path=file_name + '.png')
        # Save dictionary results
        with open(file_name + '.json', 'w') as f:
            json.dump(clustering_output, f, ensure_ascii=False)


class NSCNetTrainer(NetworkTrainer):

    def __init__(self, result_dir='train/results'):
        super().__init__('NSCNet', result_dir)

    def train(self, K, cluster_args, inputs, method):
        nscnet = NSCNet(config.INPUT_SHAPE, cluster_args)
        nscnet.train_model(inputs)
        clustering_output, features = nscnet.compute_clusters(inputs)
        self.save_training_results(K, clustering_output, features, method)


class VAENetTrainer(NetworkTrainer):

    def __init__(self, result_dir='train/results'):
        super().__init__('VAENet', result_dir)

    def train(self, K, cluster_args, inputs, method):
        # TODO: init VAE architecture and train it
        pass


class BASENetTrainer(NetworkTrainer):

    def __init__(self, result_dir='train/results'):
        super().__init__('BASENet', result_dir)

    def train(self, K, cluster_args, inputs, method):
        # TODO: init Baseline architecture and train it
        pass
