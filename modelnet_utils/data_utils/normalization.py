from __future__ import division
import numpy as np
import random


def length_normalize(data_points, length, warning=False):
    assert len(data_points.shape) == 2
    actual_length = len(data_points)
    if warning and actual_length != length:
        print("WARNING: the received data length does not match the expected length.")
    if actual_length > length:
        selected_idxs = np.random.choice(range(actual_length), length, replace=False)
        return data_points[selected_idxs]
    elif actual_length < length:
        padding_length = length - actual_length
        padding_idxs = np.random.choice(range(actual_length), padding_length)
        padding_points = data_points[padding_idxs]
        data_points = np.concatenate([data_points, padding_points], axis=0)
    return data_points


def feature_normalize(features, method):
    assert len(features.shape) == 2
    if method == 'L2':
        m = np.expand_dims(np.sqrt(np.sum(features ** 2, axis=1)), axis=-1)
        features /= m
    elif method == '-1~1':
        features -= np.min(features)
        features /= np.max(features)
        features = (features - 0.5) * 2.
    elif method == '0~1':
        features -= np.min(features)
        features /= np.max(features)
    elif method == 'max_1':
        m = np.expand_dims(np.max(np.abs(features), axis=1), axis=-1)
        features /= m
    elif method == '255':
        features /= 255.0
    elif method == 'channel_std':
        features -= np.mean(features, axis=0)
        features /= np.std(features, axis=0)
    elif method == 'global_std':
        features -= np.mean(features)
        features /= np.std(features)
    elif method == None:
        features = features
    else:
        raise ValueError("Unsupported normalization method: {}".format(method))
    return features


def coor_normalize(coors):
    assert len(coors.shape) == 2
    coors_min = np.min(coors, axis=0)
    coors_max = np.max(coors, axis=0)
    coors_center = (coors_min + coors_max) / 2.
    coors -= coors_center
    m = np.max(np.abs(coors))
    coors /= m
    return coors


def convert_threejs_coors(coors):
    assert len(coors.shape) == 2
    threejs_coors = np.zeros(shape=coors.shape)
    threejs_coors[:, 0] = coors[:, 1]
    threejs_coors[:, 1] = coors[:, 2]
    threejs_coors[:, 2] = coors[:, 0]
    return threejs_coors