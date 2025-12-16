## Prerequisites
### Dependencies
```
sudo apt install libeigen3-dev libboost-all-dev libjsoncpp-dev libopengl-dev mesa-utils libglfw3-dev libglm-dev
```

For detailed dependency versions and installation instructions (including OpenCV with CUDA and LibTorch),   
please refer to the Photo-SLAM [Dependencies](https://github.com/HuajianUP/Photo-SLAM/tree/main?tab=readme-ov-file#dependencies) section.

> We conducted tests on a desktop equipped with an NVIDIA RTX 4080 SUPER and running Ubuntu 20.04 LTS.

### Installation
```
git clone https://github.com/NAIL-HNU/MemGS.git

./build.sh
```

## Prepare the datasets
The benchmark datasets used in our paper include [Replica (NICE-SLAM version)](https://github.com/cvg/nice-slam) and [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download).

```
cd scripts
chmod +x ./*.sh

./download_replica.sh
./download_tum.sh
```

## Examples on Benchmark Datasets
For testing, you can use the commands below to run the system after specifying `PATH_TO_DATASET` and `PATH_TO_SAVE_RESULTS`. 
To enable the viewer, simply remove the `no_viewer` flag at the end of the command.

We provide example commands for the two datasets below:

1. On Replica RGB-D:
```
./bin/tum_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    ./cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    PATH_TO_Dataset/office0 \
    PATH_TO_SAVE_RESULTS
    # no_viewer 
```

> For Mono and RGB-D examples for each sequence of the Replica dataset,  
> please refer to `replica_mono.yaml` and `replica_rgbd.yaml` in the configuration folder.

2. On TUM RGB-D:
```
./bin/tum_mono \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_desk.yaml \
    ./cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg1_desk.yaml \
    PATH_TO_Dataset/tum_freiburg1_desk \
    ./cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_desk.txt \
    PATH_TO_SAVE_RESULTS
    # no_viewer 
```

> For Mono examples for each sequence of the TUM dataset,   
> please remove the association file in the configuration folder.

Remarks: The codebase has been further optimized since the paper submission, mainly focusing on reducing the number of points during merging. Consequently, results obtained using the released implementation may differ slightly from those reported in the paper. For reference, please consider the results obtained from running the code on your target device as the definitive ones.

## Examples with Real Cameras 
We provide an example using the Intel RealSense D455 in `examples/realsense_rgbd.cpp`.  
Please refer to `scripts/realsense_d455.sh` for running this example.

## Acknowledgement
This project is built upon the excellent work of many open-source projects.  
We would like to express our sincere gratitude to the authors of the following repositories:
- [Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM.git)
- [diff-gaussian-rasterization-w-pose](https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git)
  
## Known Issues
- The parameter `Grid.size` in the configuration file may affect the mapping quality across different scenes, even when the same dataset is run multiple times due to random initialization. We suggest that users try smaller grid size settings for better results.
- The parameter `Merge.resolution` and `Merge.interval` in the configuration file may affect the final number of points in the reconstructed model as well as GPU memory usage. We recommend setting these parameters according to the available memory capacity of the target platform:  
    - a. `Merge.resolution`: in general, higher values result in fewer points after merging, but may reduce computational efficiency.
    - b. `Merge.interval`: smaller values lead to more frequent merging operations and can be configured in conjunction with the parameters described below. 
- The parameter `Merge.enable_LBFGS_opt` in the configuration file controls whether LBFGS optimization is enabled during online merging. Enabling this option may improve mapping quality, but it also increases the computational cost of the merging process. Users may disable it and use only the initial (non-optimized) values instead of the optimized values to achieve faster runtime performance if needed.
    - a. `true`: enables LBFGS optimization during merging. To maintain real-time performance, it is recommended to use a larger `Merge.interval` value (e.g., 1000).
    - b. `false`: disables LBFGS optimization during merging. In this case, a smaller `Merge.interval` value (e.g., 100) can be used to perform more frequent merging operations.

## License
The source code of this package is released under the GPLv3 license (see the LICENSE file for details).  
The following third-party libraries are used in this package. Please refer to their respective licenses for details.