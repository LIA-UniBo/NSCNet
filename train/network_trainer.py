import os
import abc
import json
import config
import itertools

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from architectures.basenet.basenet import BaseNet
from architectures.nscnet.nscnet import NSCNet
from architectures.vaenet.vaenet import VAENet
from architectures.common.visualizer import visualize_clusters

# import matplotlib

# matplotlib.use('Agg')


class NetworkTrainer:

    def __init__(self, network_name, result_dir):
        self.network_name = network_name
        self.results_dir = result_dir

    def dbscan(self, inputs):
        method = 'dbscan'

        config_list = list(itertools.product(config.EPS_VALUES, config.MIN_SAMPLES, config.N_POSSIBLE_CLUSTERS))
        for eps, min_samples, K in config_list:
            print('*' * 40)
            print(f'CLUSTERING METHOD: {method}')
            print(f'eps: {eps} - min_samples: {min_samples}')
            print('*' * 40)

            cluster_args = {
                'config': {
                    "eps": eps,
                    "min_samples": min_samples,
                    "metric": "euclidean",
                    "compute_scores": False,
                    "n_clusters": K,
                },
                "n_clusters": K,
                "method": method
            }

            self.train(cluster_args, inputs, method)

        # Save plots
        self._save_dbscan_training_plots(method)

        print('Training completed.\n\n')

    def kmeans(self, inputs):
        method = 'kmeans'

        for K in config.N_POSSIBLE_CLUSTERS:
            print('*' * 40)
            print(f'CLUSTERING METHOD: {method}')
            print(f'k: {K}')
            print('*' * 40)

            cluster_args = {
                'config': {
                    "n_clusters": K,
                    "compute_scores": False
                },
                "n_clusters": K,
                "method": method
            }

            self.train(cluster_args, inputs, method)

        # Save plots
        self._save_kmeans_training_plots(method)

        print('Training completed.\n\n')

    @abc.abstractmethod
    def train(self, **kwargs):
        return

    def _save_training_results(self, cluster_args, clustering_output, features, clustering_method):

        # Save clustering parameters along with the results received from the clustering algorithm
        clustering_output.update(cluster_args)

        # Adapt the content of the clustering_output so it can be saved as a json (readability purpose)
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
        silhouette_sample_scores = []
        cluster_labels = []

        for file in os.listdir(self.results_dir):
            if file.startswith(f'{self.network_name}_{clustering_method_name}') and file.endswith('json'):
                with open(os.path.join(self.results_dir, file), 'r') as f:
                    json_dic = json.load(f)
                    k.append(json_dic['n_clusters'])
                    silhouette.append(json_dic['silhouette'])
                    eps.append(json_dic['eps'])
                    min_samples.append(json_dic['min_samples'])
                    silhouette_sample_scores.append(json_dic['silhouette_sample_scores'])
                    cluster_labels.append(json_dic['labels'])

        # Sort values by k, eps and min_samples
        k, silhouette_k = zip(*sorted(zip(k, silhouette)))
        eps, silhouette_eps = zip(*sorted(zip(eps, silhouette)))
        min_samples, silhouette_min_samples = zip(*sorted(zip(min_samples, silhouette)))

        dbscan_clusters = [len(np.unique(dbscan_clusters)) for dbscan_clusters in cluster_labels]


        # Create the plots
        fig = plt.figure(figsize=(10, 10))

        ax1 = fig.add_subplot(4, 1, 1)
        ax1.title.set_text('K - SILHOUETTE')
        ax1.plot(k, silhouette_k)
        ax1.set_xlabel('K')
        ax1.set_ylabel('SILHOUETTE')

        ax2 = fig.add_subplot(4, 1, 2)
        ax2.title.set_text('EPS - SILOHUETTE')
        ax2.plot(eps, silhouette_eps)
        ax2.set_xlabel('EPS')
        ax2.set_ylabel('SILHOUETTE')

        ax3 = fig.add_subplot(4, 1, 3)
        ax3.title.set_text('MIN_SAMPLES - SILOHUETTE')
        ax3.plot(min_samples, silhouette_min_samples)
        ax3.set_xlabel('MIN_SAMPLES')
        ax3.set_ylabel('SILHOUETTE')

        ax4 = fig.add_subplot(4, 1, 4)
        ax4.title.set_text('K - DBSCAN CLUSTERS')
        ax4.plot(k, dbscan_clusters)
        ax4.set_xlabel('K')
        ax4.set_ylabel('DBSCAN CLUSTERS')

        fig.subplots_adjust(hspace=0.5)

        # Save to file system
        file_name = f'{self.network_name}_{clustering_method_name}_PLOTS.png'
        file_path = os.path.join(self.results_dir, file_name)
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)

        self._create_silhouette_samples_plot(cluster_labels,
                                             silhouette_sample_scores,
                                             clustering_method_name,
                                             k,
                                             silhouette)

    def _save_kmeans_training_plots(self, clustering_method_name):

        k = []
        silhouette = []
        inertia = []
        silhouette_sample_scores = []
        cluster_labels = []

        for file in os.listdir(self.results_dir):
            if file.startswith(f'{self.network_name}_{clustering_method_name}') and file.endswith('json'):
                with open(os.path.join(self.results_dir, file), 'r') as f:
                    json_dic = json.load(f)
                    k.append(json_dic['n_clusters'])
                    silhouette.append(json_dic['silhouette'])
                    inertia.append(json_dic['inertia'])
                    silhouette_sample_scores.append(json_dic['silhouette_sample_scores'])
                    cluster_labels.append(json_dic['labels'])

        # Sort by K
        k, inertia, silhouette, silhouette_sample_scores, cluster_labels = zip(
            *sorted(zip(k, inertia, silhouette, silhouette_sample_scores, cluster_labels)))

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

        self._create_silhouette_samples_plot(cluster_labels,
                                             silhouette_sample_scores,
                                             clustering_method_name,
                                             k,
                                             silhouette)

    def _create_silhouette_samples_plot(self, cluster_labels, silhouette_sample_scores, clustering_method_name,
                                        k, silhouette):
        fig, subplots = plt.subplots(len(k), 1, gridspec_kw={'height_ratios': k}, figsize=(6.5, 0.8 * sum(k)))
        for idx, n_current_cluster in enumerate(k):

            subplot = subplots[idx]

            ith_sample_silhouette_values = np.asarray(silhouette_sample_scores[idx])
            ith_cluster_labels = np.asarray(cluster_labels[idx])
            silhouette_avg = np.asarray(silhouette[idx])

            y_lower = 10
            cluster_sizes = []
            for n in range(n_current_cluster):
                # Aggregate the silhouette scores for samples belonging to cluster n_current_cluster, and sort them
                ith_cluster_silhouette_values = ith_sample_silhouette_values[ith_cluster_labels == n]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(n) / n_current_cluster)
                subplot.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                subplot.text(-0.05, y_lower + 0.5 * size_cluster_i, str(n))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10

                cluster_sizes.append(size_cluster_i)

            # print(f'variance with {n_current_cluster}
            # clusters is: {np.var(cluster_sizes)}, silhouette_index: {silhouette_avg}')

            subplot.set_title("SILHOUETTE SAMPLES")
            subplot.set_xlabel("Silhouette")
            subplot.set_ylabel("Cluster")

            # Draw the average silhouette
            subplot.axvline(x=silhouette_avg, color="red", linestyle="--")

            # Set axis ticks
            subplot.set_yticks([])
            subplot.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        # Reduce vertical spacing between subplots
        fig.subplots_adjust(hspace=0.05)
        # Save to file system
        file_name = f'{self.network_name}_{clustering_method_name}_SILHOUETTE_PLOTS.png'
        file_path = os.path.join(self.results_dir, file_name)
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)


class NSCNetTrainer(NetworkTrainer):

    def __init__(self, result_dir='train/results'):
        super().__init__('NSCNet', result_dir)

    def train(self, cluster_args, inputs, method):
        nscnet = NSCNet(config.INPUT_SHAPE, cluster_args)
        nscnet.train_model(inputs)

        nscnet.cluster_args['compute_scores'] = True
        clustering_output, features = nscnet.compute_clusters(inputs)
        self._save_training_results(cluster_args['config'], clustering_output, features, method)


class VAENetTrainer(NetworkTrainer):

    def __init__(self, result_dir='train/results'):
        super().__init__('VAENet', result_dir)

    def train(self, cluster_args, inputs, method):
        vaenet = VAENet(config.INPUT_SHAPE, cluster_args)
        vaenet.train_model(inputs)

        vaenet.cluster_args['compute_scores'] = True
        clustering_output, features = vaenet.compute_clusters(inputs)
        self._save_training_results(cluster_args['config'], clustering_output, features, method)


class BASENetTrainer(NetworkTrainer):

    def __init__(self, result_dir='train/results'):
        super().__init__('BASENet', result_dir)

    def train(self, cluster_args, inputs, method):

        basenet = BaseNet(cluster_args)

        basenet.cluster_args['compute_scores'] = True
        clustering_output, features = basenet.compute_clusters(inputs)
        self._save_training_results(cluster_args['config'], clustering_output, features, method)
