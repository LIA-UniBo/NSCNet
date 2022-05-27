import os
import time
import config

from train.network_trainer import BASENetTrainer, NSCNetTrainer, VAENetTrainer, VADENetTrainer
from architectures.common.images_loader import import_image_np_dataset, import_image_tf_dataset


def create_required_folders():
    os.makedirs('data', exist_ok=True)
    os.makedirs('train/results', exist_ok=True)


def create_inputs(tf_dataset, dummy_dataset=False):
    start_time = time.time()
    print('inputs creation started...')

    if tf_dataset:
        inputs = import_image_tf_dataset(config.IMAGES_PATH,
                                        32,
                                        (config.INPUT_SHAPE[0], config.INPUT_SHAPE[1]),
                                        shuffle=True,
                                        rgb_normalize=True)
    else:
        inputs = import_image_np_dataset(config.IMAGES_PATH,
                                         (config.INPUT_SHAPE[0], config.INPUT_SHAPE[1]),
                                         config.RGB_NORMALIZATION)
    if dummy_dataset:
        print('using dummy dataset')
        inputs = inputs[:500]

    execution_time = time.time() - start_time
    print("inputs creation completed in {} seconds.".format(round(execution_time, 2)))

    return inputs


def vaenet(train_only=False, debug=False):
    inputs = init(tf_dataset=True)
    vaenet_trainer = VAENetTrainer(train_only=train_only, debug=debug)
    vaenet_trainer.kmeans(inputs)
    vaenet_trainer.dbscan(inputs)

def vadenet(debug=False):
    inputs = init(tf_dataset=True)
    vadenet_trainer = VADENetTrainer(debug=debug)
    vadenet_trainer.gaussian_mixture(inputs, auto=True)
    #vadenet_trainer.kmeans(inputs)
    #vadenet_trainer.gaussian_mixture(inputs, auto=False)


def nscnet():
    inputs = init()
    nscnet_trainer = NSCNetTrainer()
    nscnet_trainer.kmeans(inputs)


def basenet():
    inputs = init()
    basenet_trainer = BASENetTrainer()
    basenet_trainer.kmeans(inputs)
    basenet_trainer.dbscan(inputs)
    basenet_trainer.gaussian_mixture(inputs)


def init(tf_dataset=False):
    create_required_folders()
    return create_inputs(tf_dataset)


if __name__ == '__main__':
    vadenet()
