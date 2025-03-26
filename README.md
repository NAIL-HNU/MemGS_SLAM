# 1. Overview
Recent advancements in 3D Gaussian Splatting
(3DGS) have made a significant impact on rendering and
reconstruction techniques. Current research predominantly focuses on improving rendering performance and reconstruction
quality using high-performance desktop GPUs, largely overlooking applications for embedded platforms like micro air
vehicles (MAVs). These devices, with their limited computational resources and memory, often face a trade-off between
system performance and reconstruction quality. In this paper,
we improve existing methods in terms of GPU memory usage
while enhancing rendering quality. Specifically, to address
redundant 3D Gaussian primitives in SLAM, we propose
merging them in voxel space based on geometric similarity. This
reduces GPU memory usage without impacting system runtime
performance. Furthermore, rendering quality is improved by
initializing 3D Gaussian primitives via Patch-Grid (PG) point
sampling, enabling more accurate modeling of the entire scene.
Quantitative and qualitative evaluations on publicly available
datasets demonstrate the effectiveness of our improvements.
# 2. Prerequisites
## 2.1 Dependencies
```
sudo apt install libeigen3-dev libboost-all-dev libjsoncpp-dev libopengl-dev mesa-utils libglfw3-dev libglm-dev
```
## 2.2 Installation
```
git clone --recursive https://github.com/NAIL-HNU/MemGS.git

./build.sh
```
# 3. Usage
## 3.1 Download the dataset
```
./scripts/download_replica.sh
./scripts/download_tum.sh
```
## 3.2 Run Replica & TUM RGB-D
```
./scripts/replica_mono.sh
./scripts/replica_rgbd.sh
./scripts/tum_mono.sh
./scripts/tum_rgbd.sh
```
## 3.3 MemGS Examples with Real Cameras
In the file `examples/realsense_rgbd.cpp`, we provide an example with the `Intel RealSense D455`

```
./scripts/realsense_d455.sh
```
# 4. Evaluation
For ease of evaluation, we also uploaded the baseline code and added code snippets for saving results.  
It is worth noting that in order to maintain consistency with the original directory structure, the locations of these startup scripts are different.   
To run these scripts, follow the steps below:
## 4.1 Photo-SLAM
```
git checkout eval/Photo_SLAM

# run scripts
./scripts/replica_mono.sh
./scripts/replica_rgbd.sh

./scripts/tum_mono.sh
./scripts/tum_rgbd.sh
```
## 4.2 MonoGS
```
git checkout eval/MonoGS

# run scripts
./scripts/replica.sh
./scripts/tum.sh
```
## 4.3 SplaTAM
```
git checkout eval/SplaTAM

# run scripts
./bash_scripts/replica.sh
./bash_scripts/tum.sh
```
## 4.4 GS_ICP_SLAM
```
git checkout eval/GS_ICP_SLAM

# run scripts
./replica.sh
./tum.sh
```
## 4.5 Ours
```
git checkout main

# run scripts
./scripts/replica_mono.sh
./scripts/replica_rgbd.sh

./scripts/tum_mono.sh
./scripts/tum_rgbd.sh
```
# Acknowledgement
We extend our gratitude to the developers of the repositories listed below for making their code available. 
- [HuajianUP/Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM.git): A real-time framework for visual odometry-based tracking and mapping of 3DGS
- [muskie82/MonoGS](https://github.com/muskie82/MonoGS.git) A multi-process framework for tracking and mapping based on 3DGS
- [spla-tam/SplaTAM](https://github.com/spla-tam/SplaTAM.git): The first open source 3DGS-based SLAM framework
- [Lab-of-AI-and-Robotics/GS_ICP_SLAM](https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM.git): A novel SLAM framework combining G-ICP and 3DGS