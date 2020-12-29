from __future__ import division, absolute_import
import os
import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# print(currentdir)
sys.path.insert(0,currentdir) 
from os.path import join
import random
import numpy as np
from numpy.linalg import multi_dot, inv
from copy import deepcopy
from data_utils.augmentation import shuffle, transform, scale, rotate, ones_padding, flip
from data_utils.normalization import feature_normalize, coor_normalize, convert_threejs_coors
from point_viz.converter import PointvizConverter
# from viz_utils.potree import potree_viz
from tqdm import tqdm
import h5py
try:
    import queue
except ImportError:
    import Queue as queue
import time
import multiprocessing


class ModelNet40(object):
    def __init__(self,
                 npoint,
                 phase,
                 batch_size,
                 aug_config=None,
                 normalization=None,
                 augmentation=False,
                 use_normal=False,
                 use_abs=True,
                 padding_channels=1,
                 queue_size=100,
                 num_worker=1,
                 home='/media/data1/Point_ConvNet'):
        # self.home = join(home, 'datasets', 'ModelNet40')
        self.home=home
        self.npoint = npoint
        self.batch_size = batch_size
        self.use_normal = use_normal
        self.phase = phase
        self.augmentation = augmentation
        self.normalization = normalization
        self.use_abs = use_abs
        self.rotate_mode = aug_config["rotate_mode"] if aug_config is not None else "u"
        self.rotate_range = aug_config["rotate_range"] if aug_config is not None else 0.
        self.scale_mode = aug_config["scale_mode"] if aug_config is not None else "u"
        self.scale_range = aug_config["scale_range"] if aug_config is not None else 0.
        self.flip = aug_config["flip"] if aug_config is not None else False
        self.shuffle = aug_config["shuffle"] if aug_config is not None else False
        self.coors = np.load(join(self.home, '{}_coors.npy'.format(phase))).astype(np.float32)
        self.normals = np.load(join(self.home, '{}_normals.npy'.format(phase))).astype(np.float32)
        self.labels = np.load(join(self.home, '{}_labels.npy'.format(phase))).astype(np.int32)
        self.feature_channels = 3 if use_normal else padding_channels
        self.data_length = len(self.labels)
        self.batch_sum = int(np.ceil(self.data_length / batch_size))
        self.idx = 0
        self.num_worker = num_worker
        self.threads = []
        self.queue_size = queue_size
        self.q = multiprocessing.Queue(maxsize=self.queue_size)
        print("==========Generator Configurations===========")
        print("Using Configurations:")
        print("Dataset home: {}".format(self.home))
        print("Phase: {}".format(self.phase))
        print("Dataset length: {}".format(self.data_length))
        print("Number of input points: {}".format(self.npoint))
        print("Batch size: {}".format(self.batch_size))
        print("Use normals: {}".format(self.use_normal))
        print("Use abs: {}".format(self.use_abs))
        print("Normalization method: {}".format(self.normalization))
        print("Augmentation: {}".format(self.augmentation))
        if self.augmentation:
            print("    Rotate range: {}, mode: {}".format(self.rotate_range, self.rotate_mode))
            print("    Scale range: {}, mode: {}".format(self.scale_range, self.scale_mode))
            print("    Flip: {}".format(self.flip))
            print("    Shuffle: {}".format(self.shuffle))
        print("Queue size: {}".format(self.queue_size))
        print("Number of threads per node: {}".format(self.num_worker))
        print("==============================================")
        self.start()

    def start(self):
        for _ in range(self.num_worker):
            thread = multiprocessing.Process(target=self.aug_preprocess)
            thread.daemon = True
            self.threads.append(thread)
            thread.start()

    def stop(self):
        for thread in self.threads:
            thread.terminate()
            thread.join()
            self.q.close()

    def aug_preprocess(self):
        # np.random.seed(int(time.time() * 1e3 - int(time.time()) * 1e3))
        np.random.seed(0)
        while True:
            if self.q.qsize() < self.queue_size:
                try:
                    batch_coors = np.zeros(shape=(self.batch_size, self.npoint, 3), dtype=np.float32)
                    batch_features = np.zeros(shape=(self.batch_size, self.npoint, self.feature_channels), dtype=np.float32)
                    batch_labels = np.zeros(shape=self.batch_size, dtype=np.int32)
                    for i in range(self.batch_size):
                        idx = np.random.randint(self.data_length)
                        coors = coor_normalize(deepcopy(self.coors[idx]))
                        features = deepcopy(self.normals[idx])
                        data = np.concatenate([coors, features], axis=-1)

                        if self.augmentation:
                            if self.shuffle:
                                data = shuffle(data)[:self.npoint, :]
                            else:
                                data = data[:self.npoint, :]
                            coors = data[:, :3]
                            features = data[:, 3:]
                            T_rotate, angle = rotate(self.rotate_range, self.rotate_mode)
                            T_scale, scale_xyz = scale(self.scale_range, self.scale_mode)
                            T_flip, flip_xy = flip(flip=self.flip)
                            T_coors = multi_dot([T_scale, T_flip, T_rotate])
                            T_features = multi_dot([inv(T_scale), T_flip, T_rotate])
                            coors = coor_normalize(transform(coors, T_coors))
                            features = transform(features, T_features)
                        if self.use_normal:
                            features = feature_normalize(features, method="L2")
                            if self.use_abs:
                                features = np.abs(features)
                            features = feature_normalize(features, method=self.normalization)
                        else:
                            features = ones_padding(coors, self.feature_channels)
                        # print(batch_coors.shape, coors.shape)
                        batch_coors[i] = coors
                        batch_features[i] = features
                        batch_labels[i] = self.labels[idx]
                    self.q.put([batch_coors, batch_features, batch_labels])
                except Exception as e: print("ERROR:", e)
            else:
                time.sleep(0.05)

    def train_generator(self):
        while True:
            if self.q.qsize() != 0:
                yield self.q.get()
            else:
                time.sleep(0.05)

    def valid_generator(self, start_idx=0):
        if start_idx is not None:
            self.idx = start_idx
        while True:
            stop_idx = int(np.min([self.idx + self.batch_size, self.data_length]))
            batch_size = stop_idx - self.idx
            batch_coors = np.zeros(shape=(self.batch_size, self.npoint, 3), dtype=np.float32)
            batch_features = np.zeros(shape=(self.batch_size, self.npoint, self.feature_channels), dtype=np.float32)
            batch_labels = np.zeros(shape=self.batch_size, dtype=np.int32)

            for i in range(batch_size):
                coors = coor_normalize(deepcopy(self.coors[self.idx])[:self.npoint, :])
                features = deepcopy(self.normals[self.idx])[:self.npoint, :]
                if self.use_normal:
                    features = feature_normalize(features, method="L2")
                    if self.use_abs:
                        features = np.abs(features)
                    features = feature_normalize(features, method=self.normalization)
                else:
                    features = ones_padding(coors, self.feature_channels)
                batch_coors[i] = coors
                batch_features[i] = features
                batch_labels[i] = self.labels[self.idx]
                self.idx += 1
            if stop_idx == self.data_length:
                self.idx = 0
            yield batch_coors, batch_features, batch_labels


if __name__ == '__main__':
    aug_config = {"rotate_mode": "g",
                  "rotate_range": np.pi,
                  "scale_mode": "g",
                  "scale_range": 0.1,
                  "flip": True,
                  "shuffle": True}

    dataset = ModelNet40(npoint=1024,
                         phase='train',
                         batch_size=16,
                         aug_config=aug_config,
                         use_normal=False,
                         augmentation=True,
                         normalization='0~1',
                         use_abs=True,
                         padding_channels=3)


    for _ in tqdm(range(500)):
        coors, features, labels = next(dataset.train_generator())
    label = labels[0]
    colors = features[0] * 255
    coors = coors[0]
    print(np.max(colors), np.min(colors))
    # potree_viz(np.concatenate([coors, colors], axis=-1), job='ModelNet', name='ModelNet_generator_train', mute=False, color=True)
    # Converter = PointvizConverter(home='/media/data1/threejs')
    # Converter.compile(task_name="ModelNet40_generator",
    #                   coors=convert_threejs_coors(coors),
    #                   default_rgb=colors)




# from __future__ import division, absolute_import
# from os.path import join
# import os
# import sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# # print(currentdir)
# sys.path.insert(0,currentdir) 
# import random
# import numpy as np
# from numpy.linalg import multi_dot, inv
# from copy import deepcopy
# from data_utils.augmentation import shuffle, transform, scale, rotate, ones_padding, flip, get_gauss_dist
# from data_utils.normalization import feature_normalize, coor_normalize
# # from viz_utils.potree import potree_viz
# from tqdm import tqdm
# import h5py
# try:
#     import queue
# except ImportError:
#     import Queue as queue
# import time
# import threading
# import multiprocessing


# class ModelNet40(object):
#     def __init__(self,
#                  npoint,
#                  phase,
#                  batch_size,
#                  rotate_setting,
#                  scale_setting,
#                  flip=False,
#                  normal=False,
#                  normalization=None,
#                  augmentation=False,
#                  gauss_drop=False,
#                  abs=True,
#                  queue_size=100,
#                  num_worker=1,
#                  home='/media/data1/G-ConvNet-lite'):
#         # self.home = join(home, 'datasets', 'ModelNet40')
#         # self.home = join(home, 'ModelNet40')
#         self.home=home
#         self.npoint = npoint
#         self.batch_size = batch_size
#         self.normal = normal
#         self.phase = phase
#         self.augmentation = augmentation
#         self.gauss_drop = gauss_drop
#         self.abs = abs
#         self.normalization = normalization
#         self.rotate_setting = rotate_setting
#         self.scale_setting = scale_setting
#         self.flip = flip
#         data = h5py.File(join(self.home, '{}.hdf5'.format(phase)), 'r')
#         self.coors = np.array(data['coors']).astype(np.float32)
#         print("init: self.coors.shape: ", self.coors.shape)
#         self.normals = np.array(data['normals']).astype(np.float32)
#         self.labels = np.array(data['labels']).astype(np.int64)
#         self.data_length = len(self.labels)
#         self.batch_sum = int(np.ceil(self.data_length / batch_size))
#         self.idx = 0
#         self.num_worker = num_worker
#         self.threads = []
#         self.queue_size = queue_size
#         self.q = multiprocessing.Queue(maxsize=self.queue_size)
#         print("INFO: Dataset from {} was loaded".format(self.home))
#         print("INFO: ModelNet40 {}ing dataset is initialized with normal={} and augmentation={}".format(self.phase, self.normal, self.augmentation))
#         print("INFO: {} normalization method is used".format(self.normalization))
#         # if self.phase == 'train':
#         self.start()

#     def start(self):
#         for _ in range(self.num_worker):
#             thread = multiprocessing.Process(target=self.aug_preprocess)
#             thread.daemon = True
#             self.threads.append(thread.daemon)
#             thread.start()


#     def aug_preprocess(self):
#         while True:
#             if self.q.qsize() < self.queue_size:
#                 sample_npoint = self.npoint if not self.gauss_drop else int(np.floor(get_gauss_dist(self.npoint, self.npoint//4)))
#                 idxs = random.sample(range(self.data_length), self.batch_size)
#                 raw_batch_coors = deepcopy(self.coors[idxs, ...])
#                 raw_batch_normals = deepcopy(self.normals[idxs, ...])
#                 raw_batch_data = np.concatenate([raw_batch_coors, raw_batch_normals], axis=-1)
#                 batch_labels = self.labels[idxs]
#                 output_batch_coors = np.zeros((self.batch_size, sample_npoint, 3), dtype=np.float32)
#                 if not self.normal:
#                     output_batch_normals = ones_padding(output_batch_coors)
#                 else:
#                     output_batch_normals = np.zeros((self.batch_size, sample_npoint, 3), dtype=np.float32)

#                 for i in range(self.batch_size):
#                     data = shuffle(raw_batch_data[i, :, :])[:sample_npoint, :]
#                     T_rotate = rotate(self.rotate_setting)
#                     T_scale = scale(self.scale_setting)
#                     T_flip = flip() if self.flip else np.eye(3)
#                     T_coors = multi_dot([T_flip, T_rotate, T_scale])
#                     T_normals = multi_dot([T_flip, T_rotate, inv(T_scale)])
#                     coors = coor_normalize(data[:, :3])
#                     normals = feature_normalize(data[:, 3:], method='L2')

#                     if self.augmentation:
#                         coors = transform(coors, T_coors)
#                         if self.normal:
#                             normals = transform(normals, T_normals)
#                             if self.abs:
#                                 normals = np.abs(feature_normalize(normals, method='L2'))
#                             else:
#                                 normals = feature_normalize(normals, method='L2')

#                     output_batch_coors[i] = coors
#                     if self.normal:
#                         output_batch_normals[i] = feature_normalize(normals, method=self.normalization)
#                 self.q.put([output_batch_normals, output_batch_coors, batch_labels])
#             else:
#                 time.sleep(0.05)

#     def train_generator(self):
#         while True:
#             if self.q.qsize() != 0:
#                 yield self.q.get()
#             else:
#                 time.sleep(0.05)

#     def test_generator(self):
#         while True:
#             stop_idx = int(np.min([self.idx + self.batch_size, self.data_length]))
#             raw_batch_coors = deepcopy(self.coors[self.idx:stop_idx, :self.npoint, :])
#             print("raw_batch_coors.shape: ", raw_batch_coors.shape, "\t npoints: ", self.npoint)
#             print("coors.shape: ", self.coors.shape)
#             raw_batch_normals = deepcopy(self.normals[self.idx:stop_idx, :self.npoint, :])

#             output_batch_coors = np.zeros((len(raw_batch_coors), self.npoint, 3), dtype=np.float32)
#             batch_labels = self.labels[self.idx:stop_idx]
#             if not self.normal:
#                 output_batch_normals = ones_padding(raw_batch_normals)
#             else:
#                 output_batch_normals = np.zeros((len(raw_batch_normals), self.npoint, 3), dtype=np.float32)

#             for i in range(len(raw_batch_normals)):
#                 if self.normal:
#                     if self.abs:
#                         normals = np.abs(feature_normalize(raw_batch_normals[i], method='L2'))
#                     else:
#                         normals = feature_normalize(raw_batch_normals[i], method='L2')
#                     output_batch_normals[i] = feature_normalize(normals, method=self.normalization)
#                 output_batch_coors[i] = coor_normalize(raw_batch_coors[i])

#             if stop_idx == self.data_length:
#                 self.idx = 0
#             else:
#                 self.idx = stop_idx
#             yield output_batch_normals, output_batch_coors, batch_labels


# if __name__ == '__main__':
#     dataset = ModelNet40(npoint=1024,
#                          phase='test',
#                          batch_size=16,
#                          normal=False,
#                          augmentation=True,
#                          gauss_drop=False,
#                          rotate_setting=[0., 0., np.pi],
#                          scale_setting=[0.1, 0.1, 0.1],
#                          normalization='0~1',
#                          abs=True)


#     label = None
#     id = 0
#     # while id != 11:
#     # while label != 0:
#     for _ in tqdm(range(10000)):
#         points, coors, labels = next(dataset.test_generator())
#         # print(np.max(points), np.min(points), np.max(coors), np.min(coors))
#         label = labels[0]
#         id += 1
#     print(id)
#     colors = points * 255
#     print(np.max(colors), np.min(colors))
#     # potree_viz(np.concatenate([coors[0], colors[0]], axis=-1), job='ModelNet', name='ModelNet_N', mute=False, color=True)