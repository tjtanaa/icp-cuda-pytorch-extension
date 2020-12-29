import os
import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandgrandparentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.insert(0,grandgrandparentdir) 
import pickle
import numpy as np
import tqdm
import det3d
import argparse
from modelnet_utils.ModelNet40_generator import ModelNet40

# from tqdm import tqdm

from point_viz.converter import PointvizConverter
# parser = argparse.ArgumentParser(description='Analayse the kitti dataset dimensions:\
#     mode: 0 := generate the numpy file ')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

# args = parser.parse_args()


if __name__ == "__main__":
    print("Current Directory: ", currentdir)
    print("Grand Grand Parent Directory: ", grandgrandparentdir)


    save_viz_path = os.path.join(currentdir, 'visualization/modelnet40')
    Converter = PointvizConverter(save_viz_path)
    # modelnet_40_dataset = ModelNet40(npoint=10000,
    #                      phase='test',
    #                      batch_size=16,
    #                      normal=False,
    #                      augmentation=False,
    #                      gauss_drop=False,
    #                      rotate_setting=[0., 0., np.pi],
    #                      scale_setting=[0.1, 0.1, 0.1],
    #                      normalization='0~1',
    #                      abs=True,
    #                      home="/media/data3/tjtanaa/ModelNet40_Tony/ModelNet40_10k")

    # aug_config = {"rotate_mode": "g",
    #               "rotate_range": np.pi,
    #               "scale_mode": "g",
    #               "scale_range": 0.1,
    #               "flip": True,
    #               "shuffle": True}
    # aug_config = {"shuffle": False}

    modelnet_40_dataset = ModelNet40(npoint=10000,
                         phase='train',
                         batch_size=16,
                         aug_config=None,
                         use_normal=False,
                         augmentation=False,
                         normalization='0~1',
                         use_abs=True,
                         padding_channels=3,
                         home="/media/data3/tjtanaa/ModelNet40_Tony/ModelNet40_10k")

    label = None
    id = 0
    # while id != 11:
    # while label != 0:
    car_id = 0
    for batch_id in tqdm.tqdm(range(modelnet_40_dataset.batch_sum)):
        coors, features,  labels = next(modelnet_40_dataset.train_generator())
        # print(np.max(points), np.min(points), np.max(coors), np.min(coors))
        label = labels[0]
        id += 1

        # print(features.shape)
        # print(coors.shape)
        # print(labels.shape)
        # print(np.unique(labels))
        # print(labels)
        # 7 is the car
        for i in range(coors.shape[0]):
            if labels[i] == 7:
                # colors = np.clip(features[i] * 255, 0 , 255)
                # print(coors[i].shape)
                # print("np.isnan(coors[i]).any(): ", np.isnan(coors[i]))
                # coors[i] += np.array([[1e-7, 1e-7, 1e-7]])
                height = coors[:,1]
                print("car_id: ", car_id)
                print("xmin: ", np.min(coors[i][:,0]), \
                        " xmax: ", np.max(coors[i][:,0]), \
                        " ymin: ", np.min(coors[i][:,1]), \
                        " ymax: ", np.max(coors[i][:,1]), \
                        " zmin: ", np.min(coors[i][:,2]), \
                        " zmax: ", np.max(coors[i][:,2])
                        )
                
                # print("10 percentile: ", np.percentile(height, 1), "\t 90 percentile: ", np.percentile(height, 99))
                # if not np.isnan(coors[i]).any():
                #     print("np.isnan(coors[i]).any(): ", np.isnan(coors[i]).any())
                bbox_params = [[0.0001,0.0001,0.0001,10,10,10,0]]
                pts_coors = coors[i]
                # pts_coors[:,1] *= -1
                # pts_coors = np.concatenate([pts_coors, np.array([[2.0,0,0]])], axis=0)
                Converter.compile("modelnet40_car_sample_{}".format(car_id), coors=pts_coors[:,[1,2,0]], bbox_params=bbox_params) #,\
                                    # defaultrgb = colors)
                car_id += 1
                # exit()
        # break

    # print(id)
    # colors = points * 255
    # print(np.max(colors), np.min(colors))





    # dataset_path = '/media/data3/tjtanaa/kitti_dataset'
    # database_path = os.path.join(dataset_path, "gt_database")
    # classes=['Car']
    # pkl_file_name = os.path.join(database_path, '%s_gt_database_level_%s.pkl' % ('train', '-'.join(classes)))

    # with open(pkl_file_name, 'rb') as f:
    #     db = pickle.load(f) 
    #     print("There are %d objects in the %s database" % (len(db), '%s_gt_database_level_%s.pkl' % ('train', '-'.join(classes))))
    #     print(len(db[0].keys()), "Keys Available: ", db[0].keys())
    #     # for keys in db: 
    #     #     print(keys, '=>', db[keys]) 
    #     #     break

    #     # save the points as [h, w, l] to be plotted as 3d points later
    #     dimensions_np = np.zeros((len(db), 3))

    #     for obj_ind in tqdm.tqdm(range(len(db))):
    #         obj = db[obj_ind]
    #         dimensions_np[obj_ind, 0] = obj['gt_box3d'][3]
    #         dimensions_np[obj_ind, 1] = obj['gt_box3d'][4]
    #         dimensions_np[obj_ind, 2] = obj['gt_box3d'][5]

    #     # save to current directory
    #     with open(os.path.join(currentdir, '%s_dimensions_np.npy' % ('-'.join(classes))), 'wb') as numpyfile:
    #         np.save(numpyfile, dimensions_np)



    # save_viz_path = os.path.join(currentdir, 'visualization')

    # Converter = PointvizConverter(save_viz_path)
    # Converter.compile("icp_kitti_sample_{}".format(1), coors=point_cloud, intensity=intensity, 
    #                 bbox_params=gt_bbox_params_list)