import os
import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# grandgrandparentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
# sys.path.insert(0,grandgrandparentdir) 
import pickle
import numpy as np
import tqdm
# import pickle
import json
import torch
import argparse
from numpy.linalg import multi_dot
from modelnet_utils.data_utils.augmentation import transform
# from tqdm import tqdm
import time

from point_viz.converter import PointvizConverter
from det3d.pc_kitti_dataset import PCKittiAugmentedDataset

from layers import *

parser = argparse.ArgumentParser(description='Preparing the template template:\
    mode: 0 := generate the numpy file ')
parser.add_argument('--test', type=int,
                    default=0,
                    help='test 0: Val_policy_1 Scale_policy_1 \n \
                          test 1: Val_policy_1 Scale_policy_2 \n \
                          test 2: Val_policy_2 Scale_policy_1 \n \
                          test 3: Val_policy_2 Scale_policy_2 \n \
                          test 4: Val_policy_3 Scale_policy_2 \n \
                          test 5: Val_policy_3 Scale_policy_2 Keep Track Rotation Matrix and Translation')
parser.add_argument('--pc_template_path', type=str,
                    default="pc_template/Car/template/",
                    help='path to the label and point cloud binary files')
parser.add_argument('--kitti_dataset_path', type=str,
                    default='/media/data3/tjtanaa/kitti_dataset',
                    help='path to the label and point cloud binary files')
# parser.add_argument('--pc_template_output_path', type=str,
#                     default="pc_template/Car/template/",
#                     help='path to store the label and the point cloud binary files')

args = parser.parse_args()


if __name__ == "__main__":
    print("Current Directory: ", currentdir)

    # pc_template_path
    pc_template_path = os.path.join(currentdir, args.pc_template_path)

    class_string = args.pc_template_path.split('/')[1]

    template_coors_np_dict = {}
    template_bbox_params_np_dict = {}
    template_bbox_params_json_dict = {}


    # load filenames of the template
    template_pc_filename = [filename for filename in os.listdir(pc_template_path) if ('pc' in filename and 'npy' in filename)]
    print("template_pc_binary_filenames: ", template_pc_filename)

    # load and store the template pc and label in dictionaries
    for idx, filename in enumerate(template_pc_filename):
        # save point cloud binary
        pc_bin_path = os.path.join(pc_template_path, "%s_template_pc_%d.npy" % (class_string, idx))
        template_coors_np = np.load(pc_bin_path)
        
        bbox_params_bin_path = os.path.join(pc_template_path, "%s_template_label_%d.npy" % (class_string, idx))
        template_bbox_params_np = np.load(bbox_params_bin_path)
        
        template_bbox_params_json = None
        # save labels in json
        with open(os.path.join(pc_template_path, "%s_template_label_%d.json" % (class_string, idx)), 'r') as f:
            lines = f.readline()
            template_bbox_params_json = json.loads(lines)
        
        # print(template_coors_np.shape)
        # print(template_bbox_params_np.shape)
        # print(template_bbox_params_json)
        template_coors_np_dict[idx] = template_coors_np
        template_bbox_params_np_dict[idx] = template_bbox_params_np
        template_bbox_params_json_dict[idx] = template_bbox_params_json


    # print(template_coors_np_dict)
    # print(template_bbox_params_np_dict)
    # print(template_bbox_params_json_dict)

    # load kitti Car database


    
    kitti_database_path = os.path.join(args.kitti_dataset_path, "gt_database")
    
    pkl_file_name = os.path.join(kitti_database_path, '%s_gt_database_level_%s.pkl' % ('train', '-'.join([class_string])))

    # note that the kitti dataset in the kitti_db is in y-down coordinate 
    # for convenience sake, reflect the y axis by multiplying y in kitti db by -1
    # the label of the bounding box is tune as appropriate
    kitti_db = None

    with open(pkl_file_name, 'rb') as f:
        kitti_db = pickle.load(f) 
        print("There are %d objects in the %s database" % (len(kitti_db), '%s_gt_database_level_%s.pkl' % ('train', '-'.join([class_string]))))
        print(len(kitti_db[0].keys()), "Keys Available: ", kitti_db[0].keys())

    if args.test == 0:

        save_viz_path = os.path.join(currentdir, 'visualization/icp_test_0/')
        Converter = PointvizConverter(save_viz_path)

        for idx, sample in enumerate(kitti_db):

            kitti_coors_np = sample['points']
            kitti_bbox_np = sample['gt_box3d']
            # print(kitti_coors_np.shape)
            # print(kitti_bbox_np.shape)
            if (kitti_coors_np.shape[0] == 0):
                continue

            kitti_bbox_params = [[ kitti_bbox_np[5],
                                    kitti_bbox_np[3],
                                    kitti_bbox_np[4],
                                    kitti_bbox_np[0],
                                    -kitti_bbox_np[1] + kitti_bbox_np[3] / 2,
                                    kitti_bbox_np[2],
                                    kitti_bbox_np[6] ]]
            kitti_coors_np[:,1] *= -1

            # centralize the kitti_coors_np by the bbox centers
            kitti_coors_np[:,:3] -= np.array(kitti_bbox_params[0][3:6]) 


            angle = -kitti_bbox_np[6] # to return it to zero
            R = np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]]) # rotation around y axis


            kitti_coors_np = transform(kitti_coors_np,  R)

            kitti_bbox_params = [[ kitti_bbox_np[5],
                                    kitti_bbox_np[3],
                                    kitti_bbox_np[4],
                                    0,
                                    0,
                                    0,
                                    0 ]]
            # print("new")
            Converter.compile("kitti_db_car_sample_{}".format(idx), coors=kitti_coors_np, bbox_params=kitti_bbox_params)


            for k, t_coors in template_coors_np_dict.items():
                torch.cuda.empty_cache()
                print("\n=> Running ICP (PyTorch) all CUDA test")
                t_coors = t_coors[:,[1,2,0]]

                # compute the single scale
                w_ratio = kitti_bbox_params[0][2] / template_bbox_params_np_dict[k][0,3]
                l_ratio = kitti_bbox_params[0][0] / template_bbox_params_np_dict[k][0,4]
                h_ratio = kitti_bbox_params[0][1] / template_bbox_params_np_dict[k][0,5]
                scale_factor = np.max([w_ratio, l_ratio, h_ratio])
                t_coors *= scale_factor

                # purturb the template_coordinates
                noise_angle = np.random.choice([-0.1, 0.1])
                noiseR = np.array([[np.cos(noise_angle), 0, np.sin(noise_angle)],
                            [0, 1, 0],
                            [-np.sin(noise_angle), 0, np.cos(noise_angle)]]) # rotation around y axis


                t_coors = transform(t_coors,  noiseR)


                start_time = time.time()
                xyz = torch.cuda.FloatTensor(t_coors)
                new_xyz = torch.cuda.FloatTensor(kitti_coors_np)
                end_time = time.time()
                torch.cuda.synchronize()
                print("    * Load data time: {}s".format(end_time - start_time))

                torch.cuda.synchronize()
                start_time = time.time()
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                # corres, cumR, src = icp_pytorch(xyz, new_xyz, 25, ratio=0.5) #.numpy().astype(int)
                corres, cumR, src = icp_pytorch(xyz, new_xyz, 100, threshold=0.0001,  ratio=0.5) #.numpy().astype(int)
                # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                torch.cuda.synchronize()
                end_time = time.time()
                print("    * ICP CUDA computation time: {}s".format(end_time - start_time))
                # correct = (xyz_labels[corres[:, 0]] == new_xyz_labels[corres[:, 1]]).sum()
                # acc = correct / corres.shape[0]
                # print("    * Matching acc: {}".format(acc))

                # icp_t_coors = transform(t_coors, cumR.cpu().numpy())
                # intensity = np.ones(len(icp_t_coors) + len(kitti_coors_np))
                # intensity[:len(icp_t_coors)] = 0.5
                # intensity[-len(kitti_coors_np):] = 1.5
                # viz_icp_coors_np = np.concatenate([icp_t_coors, kitti_coors_np], axis=0)

                intensity = np.ones(len(t_coors) + len(kitti_coors_np))
                intensity[:len(t_coors)] = 0.5
                intensity[-len(kitti_coors_np):] = 1.5
                # viz_icp_coors_np = np.concatenate([t_coors, src.cpu().numpy()], axis=0)
                viz_icp_coors_np = np.concatenate([src.cpu().numpy(), kitti_coors_np], axis=0)
                # viz_icp_coors_np = np.concatenate([t_coors, kitti_coors_np], axis=0)

                Converter.compile("icp_car_sample_{}_template_{}".format(idx, k), coors=viz_icp_coors_np, intensity=intensity, bbox_params=kitti_bbox_params)
            # exit()

    if args.test == 1:

        save_viz_path = os.path.join(currentdir, 'visualization/icp_test_{}/'.format(args.test))
        Converter = PointvizConverter(save_viz_path)


        for idx, sample in enumerate(kitti_db):

            kitti_coors_np = sample['points']
            kitti_bbox_np = sample['gt_box3d']
            # print(kitti_coors_np.shape)
            # print(kitti_bbox_np.shape)
            if (kitti_coors_np.shape[0] == 0):
                continue

            kitti_bbox_params = [[ kitti_bbox_np[5],
                                    kitti_bbox_np[3],
                                    kitti_bbox_np[4],
                                    kitti_bbox_np[0],
                                    -kitti_bbox_np[1] + kitti_bbox_np[3] / 2,
                                    kitti_bbox_np[2],
                                    kitti_bbox_np[6] ]]
            kitti_coors_np[:,1] *= -1

            # centralize the kitti_coors_np by the bbox centers
            kitti_coors_np[:,:3] -= np.array(kitti_bbox_params[0][3:6]) 


            angle = -kitti_bbox_np[6] # to return it to zero
            R = np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]]) # rotation around y axis


            kitti_coors_np = transform(kitti_coors_np,  R)

            kitti_bbox_params = [[ kitti_bbox_np[5],
                                    kitti_bbox_np[3],
                                    kitti_bbox_np[4],
                                    0,
                                    0,
                                    0,
                                    0 ]]
            # print("new")
            Converter.compile("kitti_db_car_sample_{}".format(idx), coors=kitti_coors_np, bbox_params=kitti_bbox_params)


            for k, t_coors in template_coors_np_dict.items():
                torch.cuda.empty_cache()
                print("\n=> Running ICP (PyTorch) all CUDA test")
                t_coors = t_coors[:,[1,2,0]]

                # compute the single scale
                w_ratio = kitti_bbox_params[0][2] / template_bbox_params_np_dict[k][0,3]
                l_ratio = kitti_bbox_params[0][0] / template_bbox_params_np_dict[k][0,4]
                h_ratio = kitti_bbox_params[0][1] / template_bbox_params_np_dict[k][0,5]
                # scale_factor = np.max([w_ratio, l_ratio, h_ratio])
                # t_coors *= scale_factor
                t_coors[:,0] *= l_ratio
                t_coors[:,1] *= h_ratio
                t_coors[:,2] *= w_ratio

                # purturb the template_coordinates
                noise_angle = np.random.choice([-0.1, 0.1])
                noiseR = np.array([[np.cos(noise_angle), 0, np.sin(noise_angle)],
                            [0, 1, 0],
                            [-np.sin(noise_angle), 0, np.cos(noise_angle)]]) # rotation around y axis


                t_coors = transform(t_coors,  noiseR)


                start_time = time.time()
                xyz = torch.cuda.FloatTensor(t_coors)
                new_xyz = torch.cuda.FloatTensor(kitti_coors_np)
                end_time = time.time()
                torch.cuda.synchronize()
                print("    * Load data time: {}s".format(end_time - start_time))

                torch.cuda.synchronize()
                start_time = time.time()
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                # corres, cumR, src = icp_pytorch(xyz, new_xyz, 25, ratio=0.5) #.numpy().astype(int)
                corres, cumR, src = icp_pytorch(xyz, new_xyz, 100, threshold=0.0001, ratio=0.5) #.numpy().astype(int)
                # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                torch.cuda.synchronize()
                end_time = time.time()
                print("    * ICP CUDA computation time: {}s".format(end_time - start_time))
                # correct = (xyz_labels[corres[:, 0]] == new_xyz_labels[corres[:, 1]]).sum()
                # acc = correct / corres.shape[0]
                # print("    * Matching acc: {}".format(acc))

                # icp_t_coors = transform(t_coors, cumR.cpu().numpy())
                # intensity = np.ones(len(icp_t_coors) + len(kitti_coors_np))
                # intensity[:len(icp_t_coors)] = 0.5
                # intensity[-len(kitti_coors_np):] = 1.5
                # viz_icp_coors_np = np.concatenate([icp_t_coors, kitti_coors_np], axis=0)

                intensity = np.ones(len(t_coors) + len(kitti_coors_np))
                intensity[:len(t_coors)] = 0.5
                intensity[-len(kitti_coors_np):] = 1.5
                # viz_icp_coors_np = np.concatenate([t_coors, src.cpu().numpy()], axis=0)
                viz_icp_coors_np = np.concatenate([src.cpu().numpy(), kitti_coors_np], axis=0)
                # viz_icp_coors_np = np.concatenate([t_coors, kitti_coors_np], axis=0)

                Converter.compile("icp_car_sample_{}_template_{}".format(idx, k), coors=viz_icp_coors_np, intensity=intensity, bbox_params=kitti_bbox_params)
            # exit()


    if args.test == 2:

        save_viz_path = os.path.join(currentdir, 'visualization/icp_test_{}/'.format(args.test))
        Converter = PointvizConverter(save_viz_path)

        # specify the partition number for each axis
        partition_size = np.array([5,10,5]) # wlh
        partition_size = partition_size[[1,2,0]]


        for idx, sample in enumerate(kitti_db):

            kitti_coors_np = sample['points']
            kitti_bbox_np = sample['gt_box3d']
            # print(kitti_coors_np.shape)
            # print(kitti_bbox_np.shape)
            if (kitti_coors_np.shape[0] == 0):
                continue

            kitti_bbox_params = [[ kitti_bbox_np[5],
                                    kitti_bbox_np[3],
                                    kitti_bbox_np[4],
                                    kitti_bbox_np[0],
                                    -kitti_bbox_np[1] + kitti_bbox_np[3] / 2,
                                    kitti_bbox_np[2],
                                    kitti_bbox_np[6] ]]
            kitti_coors_np[:,1] *= -1

            # centralize the kitti_coors_np by the bbox centers
            kitti_coors_np[:,:3] -= np.array(kitti_bbox_params[0][3:6]) 


            angle = -kitti_bbox_np[6] # to return it to zero
            R = np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]]) # rotation around y axis


            kitti_coors_np = transform(kitti_coors_np,  R)

            kitti_bbox_params = [[ kitti_bbox_np[5],
                                    kitti_bbox_np[3],
                                    kitti_bbox_np[4],
                                    0,
                                    0,
                                    0,
                                    0 ]]
            # print("new")
            Converter.compile("kitti_db_car_sample_{}".format(idx), coors=kitti_coors_np, bbox_params=kitti_bbox_params)


            for k, t_coors in template_coors_np_dict.items():
                # torch.cuda.empty_cache()
                # print("\n=> Running ICP (PyTorch) all CUDA test")
                t_coors = t_coors[:,[1,2,0]]

                # compute the single scale
                w_ratio = kitti_bbox_params[0][2] / template_bbox_params_np_dict[k][0,3]
                l_ratio = kitti_bbox_params[0][0] / template_bbox_params_np_dict[k][0,4]
                h_ratio = kitti_bbox_params[0][1] / template_bbox_params_np_dict[k][0,5]
                scale_factor = np.max([w_ratio, l_ratio, h_ratio])
                t_coors *= scale_factor
                # t_coors[:,0] *= l_ratio
                # t_coors[:,1] *= h_ratio
                # t_coors[:,2] *= w_ratio

                t_bbox_min_y = -(template_bbox_params_np_dict[k][0,4] * scale_factor) / 2
                t_bbox_min_z = -(template_bbox_params_np_dict[k][0,5] * scale_factor) / 2
                t_bbox_min_x = -(template_bbox_params_np_dict[k][0,3] * scale_factor) / 2

                t_bbox_min = np.array([t_bbox_min_y, t_bbox_min_z, t_bbox_min_x])

                t_coors_offset = t_coors - t_bbox_min

                kitti_bbox_min_y = -kitti_bbox_params[0][0] / 2
                kitti_bbox_min_z = -kitti_bbox_params[0][1] / 2
                kitti_bbox_min_x = -kitti_bbox_params[0][2] / 2
                kitti_bbox_min = np.array([kitti_bbox_min_y, kitti_bbox_min_z, kitti_bbox_min_x])
                kitti_coors_np_offset = kitti_coors_np - kitti_bbox_min

                # compute the voxel size of template and kitti dataset
                t_voxel_size_y = (template_bbox_params_np_dict[k][0,4] * scale_factor) / partition_size[0]
                t_voxel_size_z = (template_bbox_params_np_dict[k][0,5] * scale_factor) / partition_size[1]
                t_voxel_size_x = (template_bbox_params_np_dict[k][0,3] * scale_factor) / partition_size[2]

                kitti_voxel_size_y = kitti_bbox_params[0][0] / partition_size[0]
                kitti_voxel_size_z = kitti_bbox_params[0][1] / partition_size[1]
                kitti_voxel_size_x = kitti_bbox_params[0][2] / partition_size[2]

                # partition

                t_indices_y = np.floor(t_coors_offset[:,0] / t_voxel_size_y)
                t_indices_x = np.floor(t_coors_offset[:,1] / t_voxel_size_z)
                t_indices_z = np.floor(t_coors_offset[:,2] / t_voxel_size_x)
                t_indices = t_indices_y + t_indices_x * partition_size[0] + t_indices_z * partition_size[0] * partition_size[1]

                kitti_indices_y = np.floor(kitti_coors_np_offset[:,0] / kitti_voxel_size_y)
                kitti_indices_x = np.floor(kitti_coors_np_offset[:,1] / kitti_voxel_size_z)
                kitti_indices_z = np.floor(kitti_coors_np_offset[:,2] / kitti_voxel_size_x)
                kitti_indices = kitti_indices_y + kitti_indices_x * partition_size[0] + kitti_indices_z * partition_size[0] * partition_size[1]


                # unique_t_indices = np.unique(t_indices)
                unique_kitti_indices = np.unique(kitti_indices)

                # filtered_kitti_
                # filtered_kitti_indices = kitti_indices in unique_t_indices

                index = np.argsort(unique_kitti_indices)
                sorted_unique_kitti_indices = unique_kitti_indices[index]
                sorted_index = np.searchsorted(sorted_unique_kitti_indices, t_indices)

                yindex = np.take(index, sorted_index, mode="clip")
                mask = sorted_unique_kitti_indices[yindex] != t_indices

                # filtered_t_indices = np.ma.array(yindex, mask=mask)
                # print("t_indices.shape: ", t_indices.shape)
                # print("filtered_t_indices.shape: ", filtered_t_indices.shape)

                filtered_t_coors = t_coors[~mask,:]
                print("filtered_t_coors.shape: ", filtered_t_coors.shape)
                # purturb the template_coordinates
                noise_angle = np.random.choice([-0.1, 0.1])
                noiseR = np.array([[np.cos(noise_angle), 0, np.sin(noise_angle)],
                            [0, 1, 0],
                            [-np.sin(noise_angle), 0, np.cos(noise_angle)]]) # rotation around y axis


                filtered_t_coors = transform(filtered_t_coors,  noiseR)


                start_time = time.time()
                xyz = torch.cuda.FloatTensor(filtered_t_coors)
                new_xyz = torch.cuda.FloatTensor(kitti_coors_np)
                end_time = time.time()
                torch.cuda.synchronize()
                print("    * Load data time: {}s".format(end_time - start_time))

                torch.cuda.synchronize()
                start_time = time.time()
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                # corres, cumR, src = icp_pytorch(xyz, new_xyz, 25, ratio=0.5) #.numpy().astype(int)
                corres, cumR, src = icp_pytorch(xyz, new_xyz, 100, threshold=0.000001, ratio=0.5) #.numpy().astype(int)
                # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                torch.cuda.synchronize()
                end_time = time.time()
                print("    * ICP CUDA computation time: {}s".format(end_time - start_time))


                # correct = (xyz_labels[corres[:, 0]] == new_xyz_labels[corres[:, 1]]).sum()
                # acc = correct / corres.shape[0]
                # print("    * Matching acc: {}".format(acc))

                # icp_t_coors = transform(t_coors, cumR.cpu().numpy())
                # intensity = np.ones(len(icp_t_coors) + len(kitti_coors_np))
                # intensity[:len(icp_t_coors)] = 0.5
                # intensity[-len(kitti_coors_np):] = 1.5
                # viz_icp_coors_np = np.concatenate([icp_t_coors, kitti_coors_np], axis=0)

                intensity = np.ones(len(filtered_t_coors) + len(kitti_coors_np))
                intensity[:len(filtered_t_coors)] = 0.5
                intensity[-len(kitti_coors_np):] = 1.5
                # viz_icp_coors_np = np.concatenate([t_coors, src.cpu().numpy()], axis=0)
                # viz_icp_coors_np = np.concatenate([filtered_t_coors, kitti_coors_np], axis=0)
                viz_icp_coors_np = np.concatenate([src.cpu().numpy(), kitti_coors_np], axis=0)

                Converter.compile("icp_car_sample_{}_template_{}".format(idx, k), coors=viz_icp_coors_np, intensity=intensity, bbox_params=kitti_bbox_params)
            exit()



    if args.test == 3:

        save_viz_path = os.path.join(currentdir, 'visualization/icp_test_{}/'.format(args.test))
        Converter = PointvizConverter(save_viz_path)

        # specify the partition number for each axis
        partition_size = np.array([5,10,5]) # wlh
        partition_size = partition_size[[1,2,0]]


        for idx, sample in enumerate(kitti_db):

            kitti_coors_np = sample['points']
            kitti_bbox_np = sample['gt_box3d']
            # print(kitti_coors_np.shape)
            # print(kitti_bbox_np.shape)
            if (kitti_coors_np.shape[0] == 0):
                continue

            kitti_bbox_params = [[ kitti_bbox_np[5],
                                    kitti_bbox_np[3],
                                    kitti_bbox_np[4],
                                    kitti_bbox_np[0],
                                    -kitti_bbox_np[1] + kitti_bbox_np[3] / 2,
                                    kitti_bbox_np[2],
                                    kitti_bbox_np[6] ]]
            kitti_coors_np[:,1] *= -1

            # centralize the kitti_coors_np by the bbox centers
            kitti_coors_np[:,:3] -= np.array(kitti_bbox_params[0][3:6]) 


            angle = -kitti_bbox_np[6] # to return it to zero
            R = np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]]) # rotation around y axis


            kitti_coors_np = transform(kitti_coors_np,  R)

            kitti_bbox_params = [[ kitti_bbox_np[5],
                                    kitti_bbox_np[3],
                                    kitti_bbox_np[4],
                                    0,
                                    0,
                                    0,
                                    0 ]]
            # print("new")
            Converter.compile("kitti_db_car_sample_{}".format(idx), coors=kitti_coors_np, bbox_params=kitti_bbox_params)


            for k, t_coors in template_coors_np_dict.items():
                # torch.cuda.empty_cache()
                # print("\n=> Running ICP (PyTorch) all CUDA test")
                t_coors = t_coors[:,[1,2,0]]

                # compute the single scale
                w_ratio = kitti_bbox_params[0][2] / template_bbox_params_np_dict[k][0,3]
                l_ratio = kitti_bbox_params[0][0] / template_bbox_params_np_dict[k][0,4]
                h_ratio = kitti_bbox_params[0][1] / template_bbox_params_np_dict[k][0,5]
                # scale_factor = np.max([w_ratio, l_ratio, h_ratio])
                # t_coors *= scale_factor
                t_coors[:,0] *= l_ratio
                t_coors[:,1] *= h_ratio
                t_coors[:,2] *= w_ratio

                t_bbox_min_y = -(template_bbox_params_np_dict[k][0,4] * l_ratio) / 2
                t_bbox_min_z = -(template_bbox_params_np_dict[k][0,5] * h_ratio) / 2
                t_bbox_min_x = -(template_bbox_params_np_dict[k][0,3] * w_ratio) / 2

                t_bbox_min = np.array([t_bbox_min_y, t_bbox_min_z, t_bbox_min_x])

                t_coors_offset = t_coors - t_bbox_min

                kitti_bbox_min_y = -kitti_bbox_params[0][0] / 2
                kitti_bbox_min_z = -kitti_bbox_params[0][1] / 2
                kitti_bbox_min_x = -kitti_bbox_params[0][2] / 2
                kitti_bbox_min = np.array([kitti_bbox_min_y, kitti_bbox_min_z, kitti_bbox_min_x])
                kitti_coors_np_offset = kitti_coors_np - kitti_bbox_min

                # compute the voxel size of template and kitti dataset
                t_voxel_size_y = (template_bbox_params_np_dict[k][0,4] * l_ratio) / partition_size[0]
                t_voxel_size_z = (template_bbox_params_np_dict[k][0,5] * h_ratio) / partition_size[1]
                t_voxel_size_x = (template_bbox_params_np_dict[k][0,3] * w_ratio) / partition_size[2]

                kitti_voxel_size_y = kitti_bbox_params[0][0] / partition_size[0]
                kitti_voxel_size_z = kitti_bbox_params[0][1] / partition_size[1]
                kitti_voxel_size_x = kitti_bbox_params[0][2] / partition_size[2]

                # partition

                t_indices_y = np.floor(t_coors_offset[:,0] / t_voxel_size_y)
                t_indices_x = np.floor(t_coors_offset[:,1] / t_voxel_size_z)
                t_indices_z = np.floor(t_coors_offset[:,2] / t_voxel_size_x)
                t_indices = t_indices_y + t_indices_x * partition_size[0] + t_indices_z * partition_size[0] * partition_size[1]

                kitti_indices_y = np.floor(kitti_coors_np_offset[:,0] / kitti_voxel_size_y)
                kitti_indices_x = np.floor(kitti_coors_np_offset[:,1] / kitti_voxel_size_z)
                kitti_indices_z = np.floor(kitti_coors_np_offset[:,2] / kitti_voxel_size_x)
                kitti_indices = kitti_indices_y + kitti_indices_x * partition_size[0] + kitti_indices_z * partition_size[0] * partition_size[1]


                # unique_t_indices = np.unique(t_indices)
                unique_kitti_indices = np.unique(kitti_indices)

                # filtered_kitti_
                # filtered_kitti_indices = kitti_indices in unique_t_indices

                index = np.argsort(unique_kitti_indices)
                sorted_unique_kitti_indices = unique_kitti_indices[index]
                sorted_index = np.searchsorted(sorted_unique_kitti_indices, t_indices)

                yindex = np.take(index, sorted_index, mode="clip")
                mask = sorted_unique_kitti_indices[yindex] != t_indices

                # filtered_t_indices = np.ma.array(yindex, mask=mask)
                # print("t_indices.shape: ", t_indices.shape)
                # print("filtered_t_indices.shape: ", filtered_t_indices.shape)

                filtered_t_coors = t_coors[~mask,:]
                print("filtered_t_coors.shape: ", filtered_t_coors.shape)
                # purturb the template_coordinates
                noise_angle = np.random.choice([-0.1, 0.1])
                noiseR = np.array([[np.cos(noise_angle), 0, np.sin(noise_angle)],
                            [0, 1, 0],
                            [-np.sin(noise_angle), 0, np.cos(noise_angle)]]) # rotation around y axis


                filtered_t_coors = transform(filtered_t_coors,  noiseR)


                start_time = time.time()
                xyz = torch.cuda.FloatTensor(filtered_t_coors)
                new_xyz = torch.cuda.FloatTensor(kitti_coors_np)
                end_time = time.time()
                torch.cuda.synchronize()
                print("    * Load data time: {}s".format(end_time - start_time))

                torch.cuda.synchronize()
                start_time = time.time()
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                # corres, cumR, src = icp_pytorch(xyz, new_xyz, 25, ratio=0.5) #.numpy().astype(int)
                corres, cumR, src = icp_pytorch(xyz, new_xyz, 100, threshold=0.000001, ratio=0.5) #.numpy().astype(int)
                # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                torch.cuda.synchronize()
                end_time = time.time()
                print("    * ICP CUDA computation time: {}s".format(end_time - start_time))


                # correct = (xyz_labels[corres[:, 0]] == new_xyz_labels[corres[:, 1]]).sum()
                # acc = correct / corres.shape[0]
                # print("    * Matching acc: {}".format(acc))

                # icp_t_coors = transform(t_coors, cumR.cpu().numpy())
                # intensity = np.ones(len(icp_t_coors) + len(kitti_coors_np))
                # intensity[:len(icp_t_coors)] = 0.5
                # intensity[-len(kitti_coors_np):] = 1.5
                # viz_icp_coors_np = np.concatenate([icp_t_coors, kitti_coors_np], axis=0)

                intensity = np.ones(len(filtered_t_coors) + len(kitti_coors_np))
                intensity[:len(filtered_t_coors)] = 0.5
                intensity[-len(kitti_coors_np):] = 1.5
                # viz_icp_coors_np = np.concatenate([t_coors, src.cpu().numpy()], axis=0)
                # viz_icp_coors_np = np.concatenate([filtered_t_coors, kitti_coors_np], axis=0)
                viz_icp_coors_np = np.concatenate([src.cpu().numpy(), kitti_coors_np], axis=0)

                Converter.compile("icp_car_sample_{}_template_{}".format(idx, k), coors=viz_icp_coors_np, intensity=intensity, bbox_params=kitti_bbox_params)
            # exit()


    if args.test == 4:

        save_viz_path = os.path.join(currentdir, 'visualization/icp_test_{}/'.format(args.test))
        Converter = PointvizConverter(save_viz_path)

        # specify the partition number for each axis
        partition_size = np.array([5,10,5]) # wlh
        partition_size = partition_size[[1,2,0]]

        # specify the ratio of num_template_points ot num_kitti_points
        ratio_t_kitti = 0.6

        for idx, sample in enumerate(kitti_db):

            kitti_coors_np = sample['points']
            kitti_bbox_np = sample['gt_box3d']
            # print(kitti_coors_np.shape)
            # print(kitti_bbox_np.shape)
            if (kitti_coors_np.shape[0] == 0):
                continue
            
            # random_indices = np.random.choice(np.arange(kitti_coors_np.shape[0]), 50)
            # kitti_coors_np = kitti_coors_np[random_indices,:]

            kitti_bbox_params = [[ kitti_bbox_np[5],
                                    kitti_bbox_np[3],
                                    kitti_bbox_np[4],
                                    kitti_bbox_np[0],
                                    -kitti_bbox_np[1] + kitti_bbox_np[3] / 2,
                                    kitti_bbox_np[2],
                                    kitti_bbox_np[6] ]]
            kitti_coors_np[:,1] *= -1

            # centralize the kitti_coors_np by the bbox centers
            kitti_coors_np[:,:3] -= np.array(kitti_bbox_params[0][3:6]) 


            angle = -kitti_bbox_np[6] # to return it to zero
            R = np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]]) # rotation around y axis


            kitti_coors_np = transform(kitti_coors_np,  R)

            kitti_bbox_params = [[ kitti_bbox_np[5],
                                    kitti_bbox_np[3],
                                    kitti_bbox_np[4],
                                    0,
                                    0,
                                    0,
                                    0 ]]
            # print("new")
            Converter.compile("kitti_db_car_sample_{}".format(idx), coors=kitti_coors_np, bbox_params=kitti_bbox_params)

            # purturb the template_coordinates
            noise_angle = np.random.choice([-0.3, 0.3])
            noiseR = np.array([[np.cos(noise_angle), 0, np.sin(noise_angle)],
                        [0, 1, 0],
                        [-np.sin(noise_angle), 0, np.cos(noise_angle)]]) # rotation around y axis

            noiseT = np.random.uniform(low=-0.5, high=0.5, size=(1,3))

            print("    Perturbation Parameters| angleT: ", angle, "\t noiseT: ", noiseT)

            for k, t_coors in template_coors_np_dict.items():
                # torch.cuda.empty_cache()
                # print("\n=> Running ICP (PyTorch) all CUDA test")
                t_coors = t_coors[:,[1,2,0]]

                # compute the single scale
                w_ratio = kitti_bbox_params[0][2] / template_bbox_params_np_dict[k][0,3]
                l_ratio = kitti_bbox_params[0][0] / template_bbox_params_np_dict[k][0,4]
                h_ratio = kitti_bbox_params[0][1] / template_bbox_params_np_dict[k][0,5]
                # scale_factor = np.max([w_ratio, l_ratio, h_ratio])
                # t_coors *= scale_factor
                t_coors[:,0] *= l_ratio
                t_coors[:,1] *= h_ratio
                t_coors[:,2] *= w_ratio

                t_bbox_min_y = -(template_bbox_params_np_dict[k][0,4] * l_ratio) / 2
                t_bbox_min_z = -(template_bbox_params_np_dict[k][0,5] * h_ratio) / 2
                t_bbox_min_x = -(template_bbox_params_np_dict[k][0,3] * w_ratio) / 2

                t_bbox_min = np.array([t_bbox_min_y, t_bbox_min_z, t_bbox_min_x])

                t_coors_offset = t_coors - t_bbox_min

                kitti_bbox_min_y = -kitti_bbox_params[0][0] / 2
                kitti_bbox_min_z = -kitti_bbox_params[0][1] / 2
                kitti_bbox_min_x = -kitti_bbox_params[0][2] / 2
                kitti_bbox_min = np.array([kitti_bbox_min_y, kitti_bbox_min_z, kitti_bbox_min_x])
                kitti_coors_np_offset = kitti_coors_np - kitti_bbox_min

                # compute the voxel size of template and kitti dataset
                t_voxel_size_y = (template_bbox_params_np_dict[k][0,4] * l_ratio) / partition_size[0]
                t_voxel_size_z = (template_bbox_params_np_dict[k][0,5] * h_ratio) / partition_size[1]
                t_voxel_size_x = (template_bbox_params_np_dict[k][0,3] * w_ratio) / partition_size[2]

                kitti_voxel_size_y = kitti_bbox_params[0][0] / partition_size[0]
                kitti_voxel_size_z = kitti_bbox_params[0][1] / partition_size[1]
                kitti_voxel_size_x = kitti_bbox_params[0][2] / partition_size[2]

                # partition

                t_indices_y = np.floor(t_coors_offset[:,0] / t_voxel_size_y)
                t_indices_x = np.floor(t_coors_offset[:,1] / t_voxel_size_z)
                t_indices_z = np.floor(t_coors_offset[:,2] / t_voxel_size_x)
                t_indices = t_indices_y + t_indices_x * partition_size[0] + t_indices_z * partition_size[0] * partition_size[1]

                kitti_indices_y = np.floor(kitti_coors_np_offset[:,0] / kitti_voxel_size_y)
                kitti_indices_x = np.floor(kitti_coors_np_offset[:,1] / kitti_voxel_size_z)
                kitti_indices_z = np.floor(kitti_coors_np_offset[:,2] / kitti_voxel_size_x)
                kitti_indices = kitti_indices_y + kitti_indices_x * partition_size[0] + kitti_indices_z * partition_size[0] * partition_size[1]


                # unique_t_indices = np.unique(t_indices)
                unique_kitti_indices = np.unique(kitti_indices)

                # filtered_kitti_
                # filtered_kitti_indices = kitti_indices in unique_t_indices

                index = np.argsort(unique_kitti_indices)
                sorted_unique_kitti_indices = unique_kitti_indices[index]
                sorted_index = np.searchsorted(sorted_unique_kitti_indices, t_indices)

                yindex = np.take(index, sorted_index, mode="clip")
                mask = sorted_unique_kitti_indices[yindex] != t_indices

                # filtered_t_indices = np.ma.array(yindex, mask=mask)
                # print("t_indices.shape: ", t_indices.shape)
                # print("filtered_t_indices.shape: ", filtered_t_indices.shape)

                filtered_t_coors = t_coors[~mask,:]
                print("filtered_t_coors.shape: ", filtered_t_coors.shape)
                filtered_t_coors = filtered_t_coors[:int(ratio_t_kitti * len(kitti_coors_np)),:]
                print("Suppress ratio filtered_t_coors.shape: ", filtered_t_coors.shape)


                # purturb the template_coordinates
                filtered_t_coors = transform(filtered_t_coors,  noiseR)

                filtered_t_coors += noiseT


                start_time = time.time()
                xyz = torch.cuda.FloatTensor(filtered_t_coors)
                new_xyz = torch.cuda.FloatTensor(kitti_coors_np)
                end_time = time.time()
                torch.cuda.synchronize()
                print("    * Load data time: {}s".format(end_time - start_time))

                torch.cuda.synchronize()
                start_time = time.time()
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                # corres, cumR, src = icp_pytorch(xyz, new_xyz, 25, ratio=0.5) #.numpy().astype(int)
                corres, cumR, src = icp_pytorch(xyz, new_xyz, 100, threshold=0.000001, ratio=0.5) #.numpy().astype(int)
                # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                torch.cuda.synchronize()
                end_time = time.time()
                print("    * ICP CUDA computation time: {}s".format(end_time - start_time))


                # correct = (xyz_labels[corres[:, 0]] == new_xyz_labels[corres[:, 1]]).sum()
                # acc = correct / corres.shape[0]
                # print("    * Matching acc: {}".format(acc))

                icp_t_coors = transform(t_coors, cumR.cpu().numpy())
                # intensity = np.ones(len(icp_t_coors) + len(kitti_coors_np))
                # intensity[:len(icp_t_coors)] = 0.5
                # intensity[-len(kitti_coors_np):] = 1.5
                # viz_icp_coors_np = np.concatenate([icp_t_coors, kitti_coors_np], axis=0)

                intensity = np.ones(2*len(filtered_t_coors) + len(kitti_coors_np))
                intensity[:len(filtered_t_coors)] = 0.5 # before
                intensity[len(filtered_t_coors):2*len(filtered_t_coors)] = 1.0 # after
                intensity[-len(kitti_coors_np):] = 1.5 # target point cloud
                # viz_icp_coors_np = np.concatenate([t_coors, src.cpu().numpy()], axis=0)
                # viz_icp_coors_np = np.concatenate([filtered_t_coors, kitti_coors_np], axis=0)
                viz_icp_coors_np = np.concatenate([filtered_t_coors, src.cpu().numpy(), kitti_coors_np], axis=0)

                Converter.compile("icp_car_sample_{}_template_{}".format(idx, k), coors=viz_icp_coors_np, intensity=intensity, bbox_params=kitti_bbox_params)
            # exit()



    if args.test == 5:

        save_viz_path = os.path.join(currentdir, 'visualization/icp_test_{}/'.format(args.test))
        Converter = PointvizConverter(save_viz_path)

        # specify the partition number for each axis
        partition_size = np.array([5,10,5]) # wlh
        partition_size = partition_size[[1,2,0]]

        # specify the ratio of num_template_points ot num_kitti_points
        ratio_t_kitti = 0.6

        for idx, sample in enumerate(kitti_db):

            kitti_coors_np = sample['points']
            kitti_bbox_np = sample['gt_box3d']
            # print(kitti_coors_np.shape)
            # print(kitti_bbox_np.shape)
            if (kitti_coors_np.shape[0] == 0):
                continue
            
            # random_indices = np.random.choice(np.arange(kitti_coors_np.shape[0]), 50)
            # kitti_coors_np = kitti_coors_np[random_indices,:]

            kitti_bbox_params = [[ kitti_bbox_np[5],
                                    kitti_bbox_np[3],
                                    kitti_bbox_np[4],
                                    kitti_bbox_np[0],
                                    -kitti_bbox_np[1] + kitti_bbox_np[3] / 2,
                                    kitti_bbox_np[2],
                                    kitti_bbox_np[6],
                                    "Magenta" ]]
            kitti_coors_np[:,1] *= -1

            # centralize the kitti_coors_np by the bbox centers
            kitti_coors_np[:,:3] -= np.array(kitti_bbox_params[0][3:6]) 


            angle = -kitti_bbox_np[6] # to return it to zero
            R = np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]]) # rotation around y axis


            kitti_coors_np = transform(kitti_coors_np,  R)

            kitti_bbox_params = [[ kitti_bbox_np[5],
                                    kitti_bbox_np[3],
                                    kitti_bbox_np[4],
                                    0,
                                    0,
                                    0,
                                    0,
                                    "Magenta"]]
            # print("new")
            Converter.compile("kitti_db_car_sample_{}".format(idx), coors=kitti_coors_np, bbox_params=kitti_bbox_params)

            # purturb the template_coordinates
            noise_angle = np.random.choice([-0.3, 0.3])
            noiseR = np.array([[np.cos(noise_angle), 0, np.sin(noise_angle)],
                        [0, 1, 0],
                        [-np.sin(noise_angle), 0, np.cos(noise_angle)]]) # rotation around y axis

            noiseT = np.random.uniform(low=-0.5, high=0.5, size=(1,3))

            print("    Perturbation Parameters| angleT: ", angle, "\t noiseT: ", noiseT)

            for k, t_coors in template_coors_np_dict.items():
                # torch.cuda.empty_cache()
                # print("\n=> Running ICP (PyTorch) all CUDA test")
                t_coors = t_coors[:,[1,2,0]]

                # compute the single scale
                w_ratio = kitti_bbox_params[0][2] / template_bbox_params_np_dict[k][0,3]
                l_ratio = kitti_bbox_params[0][0] / template_bbox_params_np_dict[k][0,4]
                h_ratio = kitti_bbox_params[0][1] / template_bbox_params_np_dict[k][0,5]
                # scale_factor = np.max([w_ratio, l_ratio, h_ratio])
                # t_coors *= scale_factor
                t_coors[:,0] *= l_ratio
                t_coors[:,1] *= h_ratio
                t_coors[:,2] *= w_ratio

                t_bbox_min_y = -(template_bbox_params_np_dict[k][0,4] * l_ratio) / 2
                t_bbox_min_z = -(template_bbox_params_np_dict[k][0,5] * h_ratio) / 2
                t_bbox_min_x = -(template_bbox_params_np_dict[k][0,3] * w_ratio) / 2

                t_bbox_min = np.array([t_bbox_min_y, t_bbox_min_z, t_bbox_min_x])

                t_coors_offset = t_coors - t_bbox_min

                kitti_bbox_min_y = -kitti_bbox_params[0][0] / 2
                kitti_bbox_min_z = -kitti_bbox_params[0][1] / 2
                kitti_bbox_min_x = -kitti_bbox_params[0][2] / 2
                kitti_bbox_min = np.array([kitti_bbox_min_y, kitti_bbox_min_z, kitti_bbox_min_x])
                kitti_coors_np_offset = kitti_coors_np - kitti_bbox_min

                # compute the voxel size of template and kitti dataset
                t_voxel_size_y = (template_bbox_params_np_dict[k][0,4] * l_ratio) / partition_size[0]
                t_voxel_size_z = (template_bbox_params_np_dict[k][0,5] * h_ratio) / partition_size[1]
                t_voxel_size_x = (template_bbox_params_np_dict[k][0,3] * w_ratio) / partition_size[2]

                kitti_voxel_size_y = kitti_bbox_params[0][0] / partition_size[0]
                kitti_voxel_size_z = kitti_bbox_params[0][1] / partition_size[1]
                kitti_voxel_size_x = kitti_bbox_params[0][2] / partition_size[2]

                # partition

                t_indices_y = np.floor(t_coors_offset[:,0] / t_voxel_size_y)
                t_indices_x = np.floor(t_coors_offset[:,1] / t_voxel_size_z)
                t_indices_z = np.floor(t_coors_offset[:,2] / t_voxel_size_x)
                t_indices = t_indices_y + t_indices_x * partition_size[0] + t_indices_z * partition_size[0] * partition_size[1]

                kitti_indices_y = np.floor(kitti_coors_np_offset[:,0] / kitti_voxel_size_y)
                kitti_indices_x = np.floor(kitti_coors_np_offset[:,1] / kitti_voxel_size_z)
                kitti_indices_z = np.floor(kitti_coors_np_offset[:,2] / kitti_voxel_size_x)
                kitti_indices = kitti_indices_y + kitti_indices_x * partition_size[0] + kitti_indices_z * partition_size[0] * partition_size[1]


                # unique_t_indices = np.unique(t_indices)
                unique_kitti_indices = np.unique(kitti_indices)

                # filtered_kitti_
                # filtered_kitti_indices = kitti_indices in unique_t_indices

                index = np.argsort(unique_kitti_indices)
                sorted_unique_kitti_indices = unique_kitti_indices[index]
                sorted_index = np.searchsorted(sorted_unique_kitti_indices, t_indices)

                yindex = np.take(index, sorted_index, mode="clip")
                mask = sorted_unique_kitti_indices[yindex] != t_indices

                # filtered_t_indices = np.ma.array(yindex, mask=mask)
                # print("t_indices.shape: ", t_indices.shape)
                # print("filtered_t_indices.shape: ", filtered_t_indices.shape)

                filtered_t_coors = t_coors[~mask,:]
                print("filtered_t_coors.shape: ", filtered_t_coors.shape)
                filtered_t_coors = filtered_t_coors[:int(ratio_t_kitti * len(kitti_coors_np)),:]
                print("Suppress ratio filtered_t_coors.shape: ", filtered_t_coors.shape)


                # purturb the template_coordinates
                filtered_t_coors = transform(filtered_t_coors,  noiseR)

                filtered_t_coors += noiseT

                t_bbox_params_coors = template_bbox_params_np_dict[k][:,[1,2,0]] + noiseT # shifted
                t_bbox_params_lhw =  template_bbox_params_np_dict[k][:,[4,5,3]] * np.array([l_ratio, h_ratio, w_ratio])
                t_bbox_params_ry = template_bbox_params_np_dict[k][:,6] + noise_angle




                start_time = time.time()
                xyz = torch.cuda.FloatTensor(filtered_t_coors)
                new_xyz = torch.cuda.FloatTensor(kitti_coors_np)
                end_time = time.time()
                torch.cuda.synchronize()
                print("    * Load data time: {}s".format(end_time - start_time))

                torch.cuda.synchronize()
                start_time = time.time()
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                # corres, cumR, src = icp_pytorch(xyz, new_xyz, 25, ratio=0.5) #.numpy().astype(int)
                corres, cumR, cumT, src = icp_pytorch(xyz, new_xyz, 100, threshold=0.000001, ratio=0.5) #.numpy().astype(int)
                # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                torch.cuda.synchronize()
                end_time = time.time()
                print("    * ICP CUDA computation time: {}s".format(end_time - start_time))


                # correct = (xyz_labels[corres[:, 0]] == new_xyz_labels[corres[:, 1]]).sum()
                # acc = correct / corres.shape[0]
                # print("    * Matching acc: {}".format(acc))

                cumR_np = cumR.cpu().numpy()
                cumT_np = cumT.cpu().numpy()
                icp_t_coors = filtered_t_coors @ cumR_np + cumT_np
                icp_t_bbox_params_coors = t_bbox_params_coors @ cumR_np + cumT_np
                icp_t_bbox_params_lhw = t_bbox_params_lhw
                icp_t_bbox_params_ry = t_bbox_params_ry + np.arctan2(-cumR_np[0,2], np.linalg.norm(cumR_np[1:,2]))
                # print("icp_t_bbox_params_ry: ", icp_t_bbox_params_ry.shape)
                # print(icp_t_bbox_params_lhw.shape)
                icp_t_bbox_params = [icp_t_bbox_params_lhw[0].tolist() + icp_t_bbox_params_coors[0].tolist() + [icp_t_bbox_params_ry[0], "Green"]]

                # print(icp_t_bbox_params)

                intensity = np.ones(3*len(filtered_t_coors) + len(kitti_coors_np))
                intensity[:len(filtered_t_coors)] = 0.5 # before
                intensity[len(filtered_t_coors):2*len(filtered_t_coors)] = 1.0 # after
                intensity[2*len(filtered_t_coors):3*len(filtered_t_coors)] = 1.5 # manually rotation
                intensity[-len(kitti_coors_np):] = 2.0 # target point cloud
                # viz_icp_coors_np = np.concatenate([t_coors, src.cpu().numpy()], axis=0)
                # viz_icp_coors_np = np.concatenate([filtered_t_coors, kitti_coors_np], axis=0)
                viz_icp_coors_np = np.concatenate([filtered_t_coors, src.cpu().numpy(), icp_t_coors, kitti_coors_np], axis=0)

                bbox_params = kitti_bbox_params + icp_t_bbox_params
                # print(bbox_params)

                Converter.compile("icp_car_sample_{}_template_{}".format(idx, k), coors=viz_icp_coors_np, intensity=intensity, bbox_params=bbox_params)
            # exit()