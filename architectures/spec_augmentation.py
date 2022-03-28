#Reference: https://arxiv.org/pdf/1904.08779.pdf

import numpy as np
import tensorflow as tf
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

POLICIES = {
"LB":{"W":80,"F":27,"m_F":1,"T":100,"p":1.0,"m_T":1},
"LD":{"W":80,"F":27,"m_F":2,"T":100,"p":1.0,"m_T":2},
"SM":{"W":40,"F":15,"m_F":2,"T":70,"p":0.2,"m_T":2},
"SS":{"W":40,"F":27,"m_F":2,"T":70,"p":0.2,"m_T":2}
}

def augment(spectrogram, **policy):

    augmented_spectrogram = np.copy(spectrogram)
    augmented_spectrogram = time_warp(augmented_spectrogram, **policy)
    augmented_spectrogram = frequency_mask(augmented_spectrogram, **policy)
    augmented_spectrogram = time_mask(augmented_spectrogram, **policy)

    return augmented_spectrogram

def time_warp(spectrogram, W, **kwargs):

    assert len(spectrogram.shape)==3

    v, tau = spectrogram.shape[0], spectrogram.shape[1]

    mid_line = spectrogram[v//2]

    random_point = mid_line[np.random.randint(W, tau - W)] # random point along the horizontal/time axis
    w = np.random.randint((-W), W) # distance

    # Source Points
    src_points = [[[v//2, random_point[0]]]]

    # Destination Points
    dest_points = [[[v//2, random_point[0] + w]]]

    spectrogram, _ = sparse_image_warp(spectrogram, src_points, dest_points, num_boundary_points=2)

    return spectrogram.numpy()

def frequency_mask(spectrogram, m_F, F, **kwargs):

    assert len(spectrogram.shape)==3

    v = spectrogram.shape[0] # no. of frequencies

    # apply m_F frequency masks to the mel spectrogram
    for i in range(m_F):
        f = np.random.randint(0, F) # [0, F)
        f0 = np.random.randint(0, v - f) # [0, v - f)
        spectrogram[f0:f0 + f, :, :] = 0

    return spectrogram

def time_mask(spectrogram, p, m_T, T ,**kwargs):

    assert len(spectrogram.shape)==3

    tau = spectrogram.shape[1] # time frames

    window = min(T, p*tau)

    # apply m_T time masks to the mel spectrogram
    for i in range(m_T):
        t = np.random.randint(0, window) # [0, T)
        t0 = np.random.randint(0, tau - t) # [0, tau - t)
        spectrogram[:, t0:t0 + t, :] = 0

    return spectrogram
