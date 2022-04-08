import os
import time
import config

import matplotlib.pyplot as plt
from train.network_trainer import NSCNetTrainer, VAENetTrainer
from architectures.images_loader import import_image_np_dataset


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
        inputs = inputs[:150]

    execution_time = time.time() - start_time
    print("inputs creation completed in {} seconds.".format(round(execution_time, 2)))

    return inputs


if __name__ == '__main__':

    create_required_folders()

    inputs = create_inputs(dummy_dataset=True)

    '''
    nscnet_trainer = NSCNetTrainer()
    nscnet_trainer.kmeans(inputs)
    nscnet_trainer.dbscan(inputs)
    '''

    vaenet_trainer = VAENetTrainer()
    # vaenet_trainer.kmeans(inputs)
    vaenet_trainer.dbscan(inputs)

