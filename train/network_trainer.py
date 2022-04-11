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


class NetworkTrainer:

    def __init__(self, network_name, results_dir):
        self.network_name = network_name
        self.results_dir = results_dir

        # Array used for saving the path of a json file that contains the result of a single network training.
        self.saved_json_file_paths = []

        # Create directories if required
        os.makedirs(self.results_dir, exist_ok=True)

    def dbscan(self, inputs):

        # Reset json paths
        self.saved_json_file_paths.clear()

        method = 'dbscan'
        file_path_prefix = self._create_file_path_prefix(method)

        config_list = list(itertools.product(config.EPS_VALUES, config.MIN_SAMPLES))

        for eps, min_samples in config_list:
            print('*' * 40)
            print(f'Training started for {self.network_name} - {method}...')
            print(f'eps: {eps} - min_samples: {min_samples}')
            print('*' * 40)

            cluster_dic = {
                'config': {
                    "eps": eps,
                    "min_samples": min_samples,
                    "metric": "euclidean",
                    "compute_scores": False,
                },
                "method": method,
                "name": f"{file_path_prefix}_eps{eps}_minsamples{min_samples}"
            }

            self.train(cluster_dic, inputs)
            print('Training complete.\n')

        # Save plots
        self._save_dbscan_training_plots(file_path_prefix)

    def kmeans(self, inputs):

        # Reset json paths
        self.saved_json_file_paths.clear()

        method = 'kmeans'
        file_path_prefix = self._create_file_path_prefix(method)

        for K in config.N_POSSIBLE_CLUSTERS:
            print('*' * 40)
            print(f'Training started for {self.network_name} - {method}...')
            print(f'k: {K}')
            print('*' * 40)

            cluster_dic = {
                'config': {
                    "n_clusters": K,
                    "compute_scores": False
                },
                "method": method,
                "name": f"{file_path_prefix}_K{K}"
            }

            self.train(cluster_dic, inputs)
            print('Training complete.\n')

        # Save plots
        self._save_kmeans_training_plots(file_path_prefix)

    @abc.abstractmethod
    def train(self, **kwargs):
        return

    def _create_file_path_prefix(self, clustering_method):
        return f'{os.path.join(self.results_dir, self.network_name)}_{clustering_method}'

    def _save_training_results(self, cluster_dic, clustering_output, features):

        file_name = cluster_dic['name']
        cluster_args = cluster_dic['config']

        # Save clustering parameters along with the results received from the clustering algorithm
        clustering_output.update(cluster_args)

        # Adapt the content of the clustering_output so it can be saved as a json (readability purpose)
        for elem in clustering_output:
            if isinstance(clustering_output[elem], np.float32):
                clustering_output[elem] = float(clustering_output[elem])
            elif isinstance(clustering_output[elem], np.ndarray):
                clustering_output[elem] = clustering_output[elem].tolist()

        # Save image showing clusters
        visualize_clusters(features, clustering_output["labels"], file_path=file_name + '.png')
        # Save dictionary results
        json_file_name = file_name + '.json'
        with open(json_file_name, 'w') as f:
            json.dump(clustering_output, f, ensure_ascii=False)

        # Add to json file path list so as to conveniently process it later
        self.saved_json_file_paths.append(file_name + '.json')

    def _save_dbscan_training_plots(self, results_dir):

        silhouette = []
        eps = []
        min_samples = []
        silhouette_sample_scores = []
        cluster_labels = []

        for file in self.saved_json_file_paths:
            with open(file, 'r') as f:
                json_dic = json.load(f)
                silhouette.append(json_dic['silhouette'])
                eps.append(json_dic['eps'])
                min_samples.append(json_dic['min_samples'])
                silhouette_sample_scores.append(json_dic['silhouette_sample_scores'])
                cluster_labels.append(json_dic['labels'])

        # Sort values by eps and min_samples
        eps, silhouette_eps = zip(*sorted(zip(eps, silhouette)))
        min_samples, silhouette_min_samples = zip(*sorted(zip(min_samples, silhouette)))

        dbscan_clusters = [len(np.unique(dbscan_clusters)) for dbscan_clusters in cluster_labels]

        # Create the plots
        fig = plt.figure(figsize=(10, 10))

        ax1 = fig.add_subplot(4, 1, 2)
        ax1.title.set_text('EPS - SILOHUETTE')
        ax1.plot(eps, silhouette_eps)
        ax1.set_xlabel('EPS')
        ax1.set_ylabel('SILHOUETTE')

        ax2 = fig.add_subplot(4, 1, 3)
        ax2.title.set_text('MIN_SAMPLES - SILOHUETTE')
        ax2.plot(min_samples, silhouette_min_samples)
        ax2.set_xlabel('MIN_SAMPLES')
        ax2.set_ylabel('SILHOUETTE')

        fig.subplots_adjust(hspace=0.5)

        # Save to file system
        file_path = f'{results_dir}_PLOTS.png'
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)

        self._create_silhouette_samples_plot(cluster_labels,
                                             silhouette_sample_scores,
                                             dbscan_clusters,
                                             silhouette,
                                             f'{results_dir}_SILHOUETTE_PLOTS.png')

    def _save_kmeans_training_plots(self, results_dir):

        k = []
        silhouette = []
        inertia = []
        silhouette_sample_scores = []
        cluster_labels = []

        for file in self.saved_json_file_paths:
            with open(file, 'r') as f:
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
        file_path = f'{results_dir}_PLOTS.png'
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)

        self._create_silhouette_samples_plot(cluster_labels,
                                             silhouette_sample_scores,
                                             k,
                                             silhouette,
                                             f'{results_dir}_SILHOUETTE_PLOTS.png')

    def _create_silhouette_samples_plot(self, cluster_labels, silhouette_sample_scores,
                                        k, silhouette, file_name):
        fig, subplots = plt.subplots(len(k), 1, gridspec_kw={'height_ratios': k}, figsize=(6.5, 0.8 * sum(k)))
        for idx, n_current_cluster in enumerate(k):

            if len(k) > 1:
                subplot = subplots[idx]
            else:
                subplot = subplots

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
        plt.savefig(file_name, bbox_inches='tight')
        plt.close(fig)


class NSCNetTrainer(NetworkTrainer):

    def __init__(self, result_dir='train/results/NSCNet'):
        super().__init__('NSCNet', result_dir)

    def train(self, cluster_dic, inputs):
        nscnet = NSCNet(config.INPUT_SHAPE, cluster_dic)
        nscnet.train_model(inputs)

        nscnet.cluster_args['compute_scores'] = True
        clustering_output, features = nscnet.compute_clusters(inputs)

        self._save_training_results(cluster_dic, clustering_output, features)

    def dbscan(self, inputs):
        print('This architecture does not support the DBSCAN algorithm')


class VAENetTrainer(NetworkTrainer):

    def __init__(self, result_dir='train/results/VAENet'):
        super().__init__('VAENet', result_dir)

    def train(self, cluster_dic, inputs):
        vaenet = VAENet(config.INPUT_SHAPE, cluster_dic)

        if not vaenet.model_already_trained:
            vaenet.train_model(inputs)
        else:
            print('__D weights already found!')

        vaenet.cluster_args['compute_scores'] = True
        clustering_output, features = vaenet.compute_clusters(inputs)

        self._save_training_results(cluster_dic, clustering_output, features)


class BASENetTrainer(NetworkTrainer):

    def __init__(self, result_dir='train/results/BASENet'):
        super().__init__('BASENet', result_dir)
        self.features = None

    def train(self, cluster_dic, inputs):

        basenet = BaseNet(cluster_dic)

        basenet.cluster_args['compute_scores'] = True
        clustering_output, self.features = basenet.compute_clusters(inputs, features=self.features)

        self._save_training_results(cluster_dic, clustering_output, self.features)



