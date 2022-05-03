
RGB_NORMALIZATION = True  # True means that images will be normalized between 0 and 1
IMAGES_PATH = "data"  # Dataset location on file system
INPUT_SHAPE = (64, 512, 3)  # Input shape of images given as input to the models
N_POSSIBLE_CLUSTERS = [64]  # k-means: possible values for the K parameter

EPS_VALUES = [0.5]  # DBSCAN: possible values for the EPS parameter
MIN_SAMPLES = [2, 4, 8, 16, 32, 64, 128]  # DBSCAN/OPTICS: possible values for the MIN_SAMPLES parameter
