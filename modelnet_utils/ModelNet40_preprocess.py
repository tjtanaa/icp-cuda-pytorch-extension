from __future__ import division
import numpy as np
import os
from tqdm import tqdm
from os.path import join
import h5py
from data_utils.os_utils import create_dir, download_zip, unzip_file


def main(phase, dataset_home, output_dir_name='ModelNet40'):
    print("INFO: processing {}ing data...".format(phase))
    output_dir = join(dataset_home, output_dir_name)
    raw_dir = join(dataset_home, 'raw/modelnet40_ply_hdf5_2048')
    output_coors = []
    output_normals = []
    output_labels = []

    h5_txt_file_list = join(raw_dir, '{}_files.txt'.format(phase))
    f = h5py.File(join(output_dir, "{}.hdf5".format(phase)), "w")
    for h5_file in tqdm(open(h5_txt_file_list)):
        filename = os.path.basename(h5_file.rstrip())
        data = h5py.File(join(raw_dir, filename))
        coors = np.array(data['data'][...])
        print("coors.shape: ", coors.shape)
        normals = np.array(data['normal'][...])
        coors[..., :] = coors[..., [0, 2, 1]]
        normals[..., :] = normals[..., [0, 2, 1]]
        output_coors.append(coors.astype(np.float32))
        output_normals.append(normals.astype(np.float32))
        output_labels.append(np.squeeze(data['label'][:]).astype(np.int64))
    output_coors = np.concatenate(output_coors, axis=0)
    output_normals = np.concatenate(output_normals, axis=0)
    output_labels = np.concatenate(output_labels, axis=0)
    f.create_dataset('coors', data=output_coors)
    f.create_dataset('normals', data=output_normals)
    f.create_dataset('labels', data=output_labels)


if __name__ == '__main__':
    # dataset_home = join(os.path.dirname(os.path.abspath(__file__)), '../../datasets')
    dataset_home = "/media/data3/tjtanaa/ModelNet40_Tony"
    output_dir_name = 'ModelNet40'
    output_dir = join(dataset_home, output_dir_name)
    # create_dir(output_dir, clean=False)
    # download_zip(download_dir=join(dataset_home, 'raw'),
    #              url='https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip')
    unzip_file(download_dir=join(dataset_home, 'raw'),
               name='modelnet40_ply_hdf5_2048')
    main('test', dataset_home=dataset_home, output_dir_name=output_dir_name)
    main('train', dataset_home=dataset_home, output_dir_name=output_dir_name)
    print("INFO: ModelNet40 preprocessing completed, data saved in: {}".format(join(dataset_home, output_dir_name))) 