import os
import abc
import json
import config
import itertools

import numpy as np
import matplotlib.pyplot as plt

from architectures.baseline import BaseNet
from architectures.nscnet import NSCNet
from architectures.vae_architecture import VAENet
from architectures.visualizer import visualize_clusters


class NetworkTrainer:

    def __init__(self, network_name, result_dir):
        self.network_name = network_name
        self.results_dir = result_dir

    def dbscan(self, inputs):
        method = 'dbscan'

        config_list = list(itertools.product(config.EPS_VALUES, config.MIN_SAMPLES, config.N_POSSIBLE_CLUSTERS))
        for eps, min_samples, K in config_list:
            cluster_args = {
                'config': {
                    "eps": eps,
                    "min_samples": min_samples,
                    "metric": "euclidean",
                    "n_clusters": K
                },
                "n_clusters": K,
                "method": method
            }

            self.train(cluster_args, inputs, method)

        # Save plots
        self._save_dbscan_training_plots(method)

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

            self.train(cluster_args, inputs, method)

        # Save plots
        self._save_kmeans_training_plots(method)

    @abc.abstractmethod
    def train(self, **kwargs):
        return

    def _save_training_results(self, cluster_args, clustering_output, features, clustering_method):

        # Save also cluster parameters
        clustering_output.update(cluster_args)

        # Adapt the content of the clustering_output so it can be saved as a json
        for elem in clustering_output:
            if isinstance(clustering_output[elem], np.float32):
                clustering_output[elem] = float(clustering_output[elem])
            elif isinstance(clustering_output[elem], np.ndarray):
                clustering_output[elem] = clustering_output[elem].tolist()

        # Create an appropriate suffix so as to help recognize the algorithm used for clustering
        if clustering_method == 'kmeans':
            suffix = f'K{cluster_args["n_clusters"]}'
        elif clustering_method == 'dbscan':
            suffix = f'K{cluster_args["n_clusters"]}_EPS{cluster_args["eps"]}_MIN_SAMPLES{cluster_args["min_samples"]}'
        else:
            suffix = 'unknown'

        file_name = f'{self.network_name}_{clustering_method}_{suffix}'
        file_name = os.path.join(self.results_dir, file_name)

        # Save image showing clusters
        visualize_clusters(features, clustering_output["labels"], file_path=file_name + '.png')
        # Save dictionary results
        with open(file_name + '.json', 'w') as f:
            json.dump(clustering_output, f, ensure_ascii=False)

    def _save_dbscan_training_plots(self, clustering_method_name):

        k = []
        silhouette = []
        eps = []
        min_samples = []

        for file in os.listdir(self.results_dir):
            if file.startswith(f'{self.network_name}_{clustering_method_name}') and file.endswith('json'):
                with open(os.path.join(self.results_dir, file), 'r') as f:
                    json_dic = json.load(f)
                    k.append(json_dic['n_clusters'])
                    silhouette.append(json_dic['silhouette'])
                    eps.append(json_dic['eps'])
                    min_samples.append(json_dic['eps'])

        # Sort by K
        k, silhouette = zip(*sorted(zip(k, silhouette)))

        # Create the plots
        fig = plt.figure(figsize=(10, 10))

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.title.set_text('K - SILHOUETTE')
        ax1.plot(k, silhouette)
        ax1.set_ylabel('SILHOUETTE')
        ax1.set_xlabel('K')

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.title.set_text('EPS - SILOHUETTE')
        ax2.plot(eps, silhouette)
        ax2.set_ylabel('SILHOUETTE')
        ax2.set_xlabel('EPS')

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.title.set_text('MIN_SAMPLES - SILOHUETTE')
        ax3.plot(min_samples, silhouette)
        ax3.set_ylabel('SILHOUETTE')
        ax3.set_xlabel('MIN_SAMPLES')

        fig.subplots_adjust(hspace=0.5)

        # Save to file system
        file_name = f'{self.network_name}_{clustering_method_name}_PLOTS.png'
        file_path = os.path.join(self.results_dir, file_name)
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)

    def _save_kmeans_training_plots(self, clustering_method_name):

        k = []
        silhouette = []
        inertia = []

        for file in os.listdir(self.results_dir):
            if file.startswith(f'{self.network_name}_{clustering_method_name}') and file.endswith('json'):
                with open(os.path.join(self.results_dir, file), 'r') as f:
                    json_dic = json.load(f)
                    k.append(json_dic['n_clusters'])
                    silhouette.append(json_dic['silhouette'])
                    inertia.append(json_dic['inertia'])

        # Sort by K
        k, inertia, silhouette = zip(*sorted(zip(k, inertia, silhouette)))

        # Create the plots
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.title.set_text('K - INERTIA')
        ax1.plot(k, inertia)
        ax1.set_ylabel('INERTIA')
        ax1.set_xlabel('K')

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.title.set_text('K - SILOHUETTE')
        ax2.plot(k, silhouette)
        ax2.set_ylabel('SILOHUETTE')
        ax2.set_xlabel('K')

        # Save to file system
        file_name = f'{self.network_name}_{clustering_method_name}_PLOTS.png'
        file_path = os.path.join(self.results_dir, file_name)
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)


class NSCNetTrainer(NetworkTrainer):

    def __init__(self, result_dir='train/results'):
        super().__init__('NSCNet', result_dir)

    def train(self, cluster_args, inputs, method):
        nscnet = NSCNet(config.INPUT_SHAPE, cluster_args)
        nscnet.train_model(inputs)
        clustering_output, features = nscnet.compute_clusters(inputs)
        self._save_training_results(cluster_args['config'], clustering_output, features, method)


class VAENetTrainer(NetworkTrainer):

    def __init__(self, result_dir='train/results'):
        super().__init__('VAENet', result_dir)

    def train(self, cluster_args, inputs, method):
        vaenet = VAENet(config.INPUT_SHAPE, cluster_args)
        vaenet.train_model(inputs)
        clustering_output, features = vaenet.compute_clusters(inputs)
        self._save_training_results(cluster_args['config'], clustering_output, features, method)


class BASENetTrainer(NetworkTrainer):

    def __init__(self, result_dir='train/results'):
        super().__init__('BASENet', result_dir)

    def train(self, cluster_args, inputs, method):

        basenet = BaseNet(cluster_args)
        clustering_output, features = basenet.clusterize(inputs)
        self._save_training_results(cluster_args['config'], clustering_output, features, method)