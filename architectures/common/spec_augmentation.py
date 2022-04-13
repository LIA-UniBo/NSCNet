# Reference: https://arxiv.org/pdf/1904.08779.pdf

# This module contains the three proposed methods to do data augmentation on spectrograms

import numpy as np
from tensorflow_addons.image import sparse_image_warp

"""
LB  : LibriSpeech basic
LD  : LibriSpeech double
SM  : Switchboard mild
SS  : Switchboard strong

W   : Time Warp parameter
F   : Frequency Mask parameter
m_F : Number of Frequency masks
T   : Time Mask parameter
p   : Parameter for calculating upper bound for time mask
m_T : Number of time masks
"""

# List of the 4 policies (set of parameters) used in the official cited paper above
POLICIES = {
    "LB": {"W": 80, "F": 27, "m_F": 1, "T": 100, "p": 1.0, "m_T": 1},
    "LD": {"W": 80, "F": 27, "m_F": 2, "T": 100, "p": 1.0, "m_T": 2},
    "SM": {"W": 40, "F": 15, "m_F": 2, "T": 70, "p": 0.2, "m_T": 2},
    "SS": {"W": 40, "F": 27, "m_F": 2, "T": 70, "p": 0.2, "m_T": 2},
    "Custom": {"W": 60, "F": 15, "m_F": 1, "T": 80, "p": 1.0, "m_T": 1}
}


def augment(spectrogram, **policy):
    # Apply all the three methods to do spec-augmentation

    augmented_spectrogram = np.copy(spectrogram)
    augmented_spectrogram = time_warp(augmented_spectrogram, **policy)
    augmented_spectrogram = frequency_mask(augmented_spectrogram, **policy)
    augmented_spectrogram = time_mask(augmented_spectrogram, **policy)

    return augmented_spectrogram


def time_warp(spectrogram, W, **kwargs):
    """
    Warp the image along the time axis
    """

    # Check that there is only one spectrogram (not a batch)
    assert len(spectrogram.shape) == 3

    # Get mel bins and time steps (height and width of the image, respectively)
    v, tau = spectrogram.shape[0], spectrogram.shape[1]

    # Take the mid horizontal line of the image
    mid_line = spectrogram[v // 2]

    # Take a random point along the horizontal/time axis
    random_point = mid_line[np.random.randint(W, tau - W)]
    # Choose warp distance
    w = np.random.randint((-W), W)

    # Source Points
    src_points = [[[v // 2, random_point[0]]]]

    # Destination Points
    dest_points = [[[v // 2, random_point[0] + w]]]

    # Warp the spectrogram
    spectrogram2, _ = sparse_image_warp(spectrogram, src_points, dest_points, num_boundary_points=2)

    return spectrogram2.numpy()


def frequency_mask(spectrogram, m_F, F, **kwargs):
    """
    Mask some random frequencies
    """

    # Check that there is only one spectrogram (not a batch)
    assert len(spectrogram.shape) == 3

    # Get mel bins (number of frequencies, corresponding the the height of the image)
    v = spectrogram.shape[0]

    # Apply m_F frequency masks to the spectrogram
    for i in range(m_F):
        f = np.random.randint(0, F)  # [0, F)
        f0 = np.random.randint(0, v - f)  # [0, v - f)
        spectrogram[f0:f0 + f, :, :] = 0

    return spectrogram


def time_mask(spectrogram, p, m_T, T, **kwargs):
    # Check that there is only one spectrogram (not a batch)
    assert len(spectrogram.shape) == 3

    # Get time steps (width of the image)
    tau = spectrogram.shape[1]

    # Fix a window with p*tau as the upper bound
    window = min(T, p * tau)

    # Apply m_T time masks to the spectrogram
    for i in range(m_T):
        t = np.random.randint(0, window)  # [0, T)
        t0 = np.random.randint(0, tau - t)  # [0, tau - t)
        spectrogram[:, t0:t0 + t, :] = 0

    return spectrogram
