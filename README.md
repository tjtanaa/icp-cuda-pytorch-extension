# ICP CUDA PyTorch Extension

CUDA Iterative Closest Point algorithm for PyTorch (>=1.3).

## Install

```
$ python setup.py build develop
```

## Benchmark

For ICP, the most time consuming part is NN search. And the CUDA version of SVD and matrix multiplication is not faster than the CPU version for small matrix. Thus I only implemented CUDA version's NN search, and leave the other parts of ICP CPU.

Because Open3D uses KDTree to accelerate NN search, the CUDA accleration is not very significant (only 2x faster).

Benchmark for 80000 points matching with single NVIDIA GTX 2080Ti and Xeon(R) CPU E5-2678 v3 @ 2.50GHz is given below. You can run the benchmark by

```
$ python ctest.py --num_points=80000
```

|      | Nearest Neighbor Search      | Iterative Closest Point      |
| ---- | ---------------------------- | ---------------------------- |
| CPU  | acc = 93.97%, time = 1.06s   | acc = 97.65%, time = 27.96ms |
| CUDA | acc = 95.54%, time = 21.3ms  | acc = 97.92%, time = 14.34ms |

## Todo

- [x] Achieve the same matching accuracy as Open3D CPU version
- [x] Accelerate CUDA ICP
- [x] Batch NN search (modified from [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d))
- [ ] Batch ICP
- [ ] Use KDTree for NN search

# __Experiments__
## __Policies to check effectiveness of the template chosen__
### Policy One

1. Compare *kitti partial point cloud with full template point cloud* in the center as origin coordinate frame and align their angle (both ry = 0)
2. Use *different scaling policies*.
3. Record the metrics between point cloud
4. Visually determine the closeness of registered point cloud

### Policy Two

1. Compare *kitti partial point cloud with partial template point cloud* in the center as origin coordinate frame and align their angle (both ry = 0)
   * Partial template point cloud is prepared as follows:
     1. Divide the kitti point cloud and template point cloud into voxels.
     2. Extract the points in the template point cloud which corresponds to the voxel index in the kitti point cloud.
2. Use *different scaling policies*.
3. Record the metrics between point cloud
4. Visually determine the closeness of registered point cloud

## __Scaling policy__
Assume that 
1. the kitti target is represented by
   * pc_kitti (point cloud of kitti)
   * bbox_gt (x_g, y_g, z_g, w_g, l_g, h_g, ry_g)

1. the template is represented by
   * pc_template (point cloud of template)
   * bbox_gt (x_t, y_t, z_t, w_t, l_t, h_t, ry_t)

### Policy One   
1. Use a single scale factor, s, where s = max( w_ratio, l_ratio, h_ratio ). w_ratio = w_g / w_t ; l_ratio = l_g / l_t; h_ratio = h_g / h_t

 
   * (x_t, y_t, z_t, w_t, l_t, h_t) * s => desired template size


2. Use 3 different scale factor for each of the directions/dimensions: w_ratio = w_g / w_t ; l_ratio = l_g / l_t; h_ratio = h_g / h_t
 
   * (x_t, w_t) * w_ratio => desired template size
   * (y_t, l_t) * l_ratio => desired template size
   * (z_t, h_t) * h_ratio => desired template size