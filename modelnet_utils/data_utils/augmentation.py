from __future__ import division

import numpy as np
from numpy.linalg import multi_dot

'''
This augmentation is performed instance-wised, not for batch-processing.

'''


def shuffle(data):
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    return data[idx, :]


def gauss_dist(mean, shift):
    if shift != 0:
        std = shift * 0.5
        return_value = np.clip(np.random.normal(mean, std), mean - shift, mean + shift)
        return return_value
    else:
        return mean

def uni_dist(mean, shift):
    if shift != 0:
        return_value = (np.random.rand() - 0.5) * 2 * shift + mean
        return return_value
    else:
        return mean


def scale(scale_range, mode='g', scale_xyz=None, T=None):
    if T is None:
        T = np.eye(3)
    if scale_xyz is None:
        if mode == 'g':
            scale_factor_x = gauss_dist(1., scale_range)
            scale_factor_y = gauss_dist(1., scale_range)
            scale_factor_z = gauss_dist(1., scale_range)
        elif mode == 'u':
            scale_factor_x = uni_dist(1., scale_range)
            scale_factor_y = uni_dist(1., scale_range)
            scale_factor_z = uni_dist(1., scale_range)
        else:
            raise ValueError("Undefined scale mode: {}".format(mode))
    else:
        scale_factor_x, scale_factor_y, scale_factor_z = scale_xyz
    T = np.dot(T, np.array([[scale_factor_x, 0, 0],
                            [0, scale_factor_y, 0],
                            [0, 0, scale_factor_z]]))
    return T, [scale_factor_x, scale_factor_y, scale_factor_z]


def flip(flip=False, flip_xy=None, T=None):
    if T is None:
        T = np.eye(3)
    if not flip:
        return np.dot(T, np.eye(3)), [1., 1.]
    else:
        if flip_xy is None:
            flip_x = -1 if np.random.rand() > 0.5 else 1
            flip_y = -1 if np.random.rand() > 0.5 else 1
        else:
            flip_x, flip_y = flip_xy
        T = np.dot(T, np.array([[flip_x, 0, 0],
                                [0, flip_y, 0],
                                [0, 0, 1]]))
        return T, [flip_x, flip_y]



def rotate(rotate_range, mode, angle=None, T=None):
    if T is None:
        T = np.eye(3)
    if angle is None:
        if mode == 'g':
            angle = gauss_dist(0., rotate_range)
        elif mode == 'u':
            angle = np.random.uniform(0., rotate_range)
        else:
            raise ValueError("Undefined rotate mode: {}".format(mode))

    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])
    T = multi_dot([R, T])
    return T, angle


def shear(shear_range, mode, shear_xy=None, T=None):
    if T is None:
        T = np.eye(3)
    # TODO: Need to change the angles_z into uniform_dist for ModelNet_40
    if shear_xy is None:
        if mode == 'g':
            lambda_x = gauss_dist(0., shear_range)
            lambda_y = gauss_dist(0., shear_range)
        elif mode == 'u':
            lambda_x = np.random.uniform(0., shear_range)
            lambda_y = np.random.uniform(0., shear_range)
        else:
            raise ValueError("Undefined shear mode: {}".format(mode))
    else:
        lambda_x, lambda_y = shear_xy
    Sx = np.array([[1, 0, lambda_x],
                   [0, 1, 0],
                   [0, 0, 1]])
    Sy = np.array([[1, 0, 0],
                   [0, 1, lambda_y],
                   [0, 0, 1]])
    T = multi_dot([Sx, Sy, T])
    return T, [lambda_x, lambda_y]


def transform(data, T):
    transformed = np.transpose(np.dot(T, np.transpose(data)))
    return transformed


def drop(data):
    raw_length = len(data)
    length = int(np.clip(-raw_length//8 * np.abs(np.random.randn()) + raw_length, raw_length-raw_length//4, raw_length))
    if length < raw_length:
        left_data = data[:length, ...]
        padding_idx = np.random.randint(length, size=[raw_length-length])
        data = np.concatenate([left_data, left_data[padding_idx, ...]], axis=0)
        return data
    else:
        return data

def ones_padding(raw_input, channels=1):
    features = np.ones(shape=(raw_input.shape[0], channels), dtype=np.float32)
    return features


