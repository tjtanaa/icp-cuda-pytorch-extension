import os
import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# grandgrandparentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
# sys.path.insert(0,grandgrandparentdir) 
import pickle
import numpy as np
import tqdm
# import det3d
import json
import argparse
from modelnet_utils.ModelNet40_generator import ModelNet40
from numpy.linalg import multi_dot
from modelnet_utils.data_utils.augmentation import transform
from pc_template.Car.orientation.orientation import orientations
# from tqdm import tqdm

from point_viz.converter import PointvizConverter
parser = argparse.ArgumentParser(description='Preparing the template template:\
    mode: 0 := generate the numpy file ')
parser.add_argument('--test', type=int,
                    default=0,
                    help='test 0: Val_policy_1 Scale_policy_1 \n \
                          test 1: <TBD> \n\
                          test 2: <TBD>')
parser.add_argument('--pc_template_path', type=str,
                    default="pc_template/Car/template/",
                    help='path to the label and point cloud binary files')
# parser.add_argument('--pc_template_output_path', type=str,
#                     default="pc_template/Car/template/",
#                     help='path to store the label and the point cloud binary files')

args = parser.parse_args()


if __name__ == "__main__":
    print("Current Directory: ", currentdir)

    # pc_template_path
    pc_template_path = os.path.join(currentdir, args.pc_template_path)
    pc_bin_list = [filename for filename in os.listdir(pc_template_path) if '.bin' in filename]
    print("List of raw point clouds: ", pc_bin_list)

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

    if args.test == 0:

        save_viz_path = os.path.join(currentdir, 'visualization/icp_test_0/')
        Converter = PointvizConverter(save_viz_path)

        # load the kitti training dataset using point pillar loader which is slow....
        

        