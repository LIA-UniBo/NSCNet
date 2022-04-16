import os
import time
import config

import matplotlib.pyplot as plt

from train.network_trainer import BASENetTrainer, NSCNetTrainer, VAENetTrainer
from architectures.common.images_loader import import_image_np_dataset


def create_required_folders():
    os.makedirs('data', exist_ok=True)
    os.makedirs('train/results', exist_ok=True)


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


def vaenet():
    inputs = init()
    vaenet_trainer = VAENetTrainer()
    vaenet_trainer.kmeans(inputs)
    vaenet_trainer.dbscan(inputs)


def nscnet():
    inputs = init()
    nscnet_trainer = NSCNetTrainer()
    nscnet_trainer.kmeans(inputs)


def basenet():
    inputs = init()
    basenet_trainer = BASENetTrainer()
    basenet_trainer.kmeans(inputs)
    # basenet_trainer.dbscan(inputs)


def init():
    create_required_folders()
    return create_inputs()


import json
from architectures.common.visualizer import visualize_clusters_distribution, visualize_clusters


def TEMP_create_distribution_plot():
    with open('NSCNet_kmeans_K256.json') as json_file:
        data = json.load(json_file)
        visualize_clusters_distribution(data['labels'], 'NSCNet_kmeans_K256.png')


def TEMP_NMI_plot():
    nmi_scores = []
    with open('NSCNet_1024_ArcFace_Logs.txt', 'r', encoding='utf8') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('NMI score: '):
                nmi_scores.append(float(line.split('NMI score: ')[1]))

    epochs = list(range(1, len(nmi_scores) + 1))

    fig = plt.figure(figsize=(13, 5))
    plt.plot(epochs, nmi_scores)
    plt.xticks(epochs)
    plt.title("NMI SCORES")
    plt.xlabel('Epochs')
    plt.ylabel('NMI')

    # plt.savefig(cluster_dic['name'] + "_nmi.png", bbox_inches='tight')
    # plt.close(fig)
    plt.show(block=True)


def TEMP_print_config_file():
    with open('architectures/nscnet/nscnet_config.py', 'r', encoding='utf8') as file:
        lines = file.readlines()
        for line in lines:
            print(repr(line + '\\')[1: -2])


def TEMP_save_results():
    dir_result = 'res_arcface_off'
    json_file_names = [
        'res_arcface_off/NSCNet_kmeans_K64.json',
        'res_arcface_off/NSCNet_kmeans_K128.json',
        'res_arcface_off/NSCNet_kmeans_K256.json',
        'res_arcface_off/NSCNet_kmeans_K512.json',
        'res_arcface_off/NSCNet_kmeans_K1024.json',
        'res_arcface_off/NSCNet_kmeans_K2048.json',
    ]
    trainer = NSCNetTrainer(dir_result)
    trainer.saved_json_file_paths = json_file_names

    trainer._save_kmeans_training_plots(dir_result)


if __name__ == '__main__':
    pass
