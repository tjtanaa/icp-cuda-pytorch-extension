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
parser.add_argument('--step', type=int,
                    default=0,
                    help='step 0: visualize the angle offseted data \n \
                          step 1: visualize the bounding boxes label of the point cloud in normalized form \n\
                          step 2: save the point cloud binary files and the labels')
parser.add_argument('--pc_template_path', type=str,
                    default="pc_template/Car/pc/",
                    help='path to the point cloud binary files')
parser.add_argument('--pc_template_output_path', type=str,
                    default="pc_template/Car/template/",
                    help='path to store the label and the point cloud binary files')

args = parser.parse_args()


if __name__ == "__main__":
    print("Current Directory: ", currentdir)

    # pc_template_path
    pc_template_path = os.path.join(currentdir, args.pc_template_path)
    pc_bin_list = [filename for filename in os.listdir(pc_template_path) if '.bin' in filename]
    print("List of raw point clouds: ", pc_bin_list)

    if args.step == 0:

        save_viz_path = os.path.join(currentdir, 'visualization/modelnet40_offset_sample/')
        Converter = PointvizConverter(save_viz_path)


        for idx, angle in enumerate(orientations):
            angle = -angle # to return it to zero
            R = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]]) # rotation around z axis

            print(angle)
            coors = np.load(os.path.join(pc_template_path, pc_bin_list[idx]))[:,:3] # ignore the rgb
            print(coors.shape)
            coors = transform(coors[:,[2,0,1]],  R)
            bbox_params = [[0.0001,0.0001,0.0001,10,10,10,0]]
            # pts_coors = coors[i]
            # # pts_coors[:,1] *= -1
            # coors = np.concatenate([coors, np.array([[0,2.0,0]])], axis=0)
            Converter.compile("modelnet40_car_offset_sample_{}".format(idx), coors=coors[:,[1,2,0]], bbox_params=bbox_params)

    if args.step == 1:    
        save_viz_path = os.path.join(currentdir, 'visualization/modelnet40_template/')
        Converter = PointvizConverter(save_viz_path)
        for idx, angle in enumerate(orientations):
            angle = -angle # to return it to zero
            R = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]]) # rotation around z axis

            # print(angle)
            coors = np.load(os.path.join(pc_template_path, pc_bin_list[idx]))[:,:3] # ignore the rgb
            # print(coors.shape)
            coors = transform(coors[:,[2,0,1]],  R)

            xmin = np.min(coors[:,0])
            xmax = np.max(coors[:,0])
            ymin = np.min(coors[:,1])
            ymax = np.max(coors[:,1])
            zmin = np.min(coors[:,2])
            zmax = np.max(coors[:,2])

            width = xmax - xmin
            length = ymax - ymin
            height = zmax - zmin
            xcenter = (xmax + xmin) / 2
            ycenter = (ymax + ymin) / 2
            zcenter = (zmax + zmin) / 2

            print("xmin: %.5f xmax: %.5f ymin: %.5f ymax: %.5f zmin: %.5f zmax: %.5f" % \
                (xmin, xmax, ymin, ymax, zmin, zmax))

            print("bbox: x: %.5f, y: %.5f, z: %.5f, w: %.5f l: %.5f h: %.5f" % \
                (xcenter, ycenter ,zcenter , width, length, height))

            bbox_params = [[length, height, width, ycenter, zcenter, xcenter, 0.0]]
            # pts_coors = coors[i]
            # # pts_coors[:,1] *= -1
            # coors = np.concatenate([coors, np.array([[0,2.0,0]])], axis=0)
            Converter.compile("car_template_{}".format(idx), coors=coors[:,[1,2,0]], bbox_params=bbox_params)
    

    if args.step == 2:    
        for idx, angle in enumerate(orientations):
            angle = -angle # to return it to zero
            R = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]]) # rotation around z axis

            # print(angle)
            coors = np.load(os.path.join(pc_template_path, pc_bin_list[idx]))[:,:3] # ignore the rgb
            # print(coors.shape)
            coors = transform(coors[:,[2,0,1]],  R)

            xmin = np.min(coors[:,0])
            xmax = np.max(coors[:,0])
            ymin = np.min(coors[:,1])
            ymax = np.max(coors[:,1])
            zmin = np.min(coors[:,2])
            zmax = np.max(coors[:,2])

            width = xmax - xmin
            length = ymax - ymin
            height = zmax - zmin
            xcenter = (xmax + xmin) / 2
            ycenter = (ymax + ymin) / 2
            zcenter = (zmax + zmin) / 2

            print("xmin: %.5f xmax: %.5f ymin: %.5f ymax: %.5f zmin: %.5f zmax: %.5f" % \
                (xmin, xmax, ymin, ymax, zmin, zmax))

            print("bbox: x: %.5f, y: %.5f, z: %.5f, w: %.5f l: %.5f h: %.5f" % \
                (xcenter, ycenter ,zcenter , width, length, height))

            bbox_params = [[length, height, width, ycenter, zcenter, xcenter, 0.0]]

            bbox_params_np = np.zeros((1,7))
            bbox_params_np[0,0] = xcenter
            bbox_params_np[0,1] = ycenter
            bbox_params_np[0,2] = zcenter
            bbox_params_np[0,3] = width
            bbox_params_np[0,4] = length
            bbox_params_np[0,5] = height
            bbox_params_np[0,6] = 0.0

            bbox_params_dict = {"x": xcenter,
                                "y": ycenter,
                                "z": zcenter,
                                "width": width,
                                "length": length,
                                "height": height,
                                "ry": 0.0}

            class_string = args.pc_template_path.split('/')[1]
            # save point cloud binary
            with open(os.path.join(args.pc_template_output_path, "%s_template_pc_%d.npy" % (class_string, idx)), 'wb') as f:
                np.save(f, coors)

            # save labels in numpy binary
            with open(os.path.join(args.pc_template_output_path, "%s_template_label_%d.npy" % (class_string, idx)), 'wb') as f:
                np.save(f, bbox_params_np)
            
            # save labels in json
            with open(os.path.join(args.pc_template_output_path, "%s_template_label_%d.json" % (class_string, idx)), 'w') as f:
                f.write(json.dumps(bbox_params_dict, ensure_ascii=False))

    


