from __future__ import division
import numpy as np
from numpy import genfromtxt
import os
from tqdm import tqdm
from os.path import join
import h5py
from data_utils.os_utils import create_dir, download_zip, unzip_file
from data_utils.normalization import length_normalize


def main(phase, dataset_home, output_dir_name='ModelNet40'):
    print("INFO: processing {}ing data...".format(phase))
    dataset_length = {"train": 9843, "test": 2468}
    output_dir = join(dataset_home, output_dir_name)
    raw_dir = join(dataset_home, 'raw/modelnet40_normal_resampled')
    category_dict = {}
    with open(join(raw_dir, "modelnet40_shape_names.txt"), "r") as f:
        for i, name in enumerate(f):
            category_dict[name[:-1]] = i
    output_coors = np.zeros(shape=(dataset_length[phase], 10000, 3), dtype=np.float32)
    output_normals = np.zeros(shape=(dataset_length[phase], 10000, 3), dtype=np.float32)
    output_labels = np.zeros(shape=dataset_length[phase], dtype=np.int32)
    with open(join(raw_dir, "modelnet40_{}.txt".format(phase)), "r") as f:
        for i, file_name in enumerate(tqdm(f, total=dataset_length[phase])):
            category = file_name.replace("_"+file_name.split("_")[-1], '')
            data = genfromtxt(join(raw_dir, category, file_name[:-1]+'.txt'), delimiter=',')
            data = length_normalize(data, length=10000, warning=True)
            output_coors[i] = data[:, :3][..., [0, 2, 1]]
            output_normals[i] = data[:, 3:][..., [0, 2, 1]]
            output_labels[i] = category_dict[category]
    np.save(join(output_dir, '{}_coors.npy'.format(phase)), output_coors)
    np.save(join(output_dir, '{}_normals.npy'.format(phase)), output_normals)
    np.save(join(output_dir, '{}_labels.npy'.format(phase)), output_labels)
    assert i+1 == dataset_length[phase], \
        "Warning: Dataset length does not match: {} vs. {}.".format(i+1, dataset_length[phase])

if __name__ == '__main__':
    # dataset_home = join(os.path.dirname(os.path.abspath(__file__)), '../../datasets')
    dataset_home = "/media/data3/tjtanaa/ModelNet40_Tony"
    output_dir_name = 'ModelNet40_10k'
    output_dir = join(dataset_home, output_dir_name)
    create_dir(output_dir, clean=False)
    # download_zip(download_dir=join(dataset_home, 'raw'),
    #              url='https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip')
    unzip_file(download_dir=join(dataset_home, 'raw'),
               name='modelnet40_normal_resampled')
    main('test', dataset_home=dataset_home, output_dir_name=output_dir_name)
    main('train', dataset_home=dataset_home, output_dir_name=output_dir_name)
    print("INFO: ModelNet40 preprocessing completed, data saved in: {}".format(join(dataset_home, output_dir_name)))


