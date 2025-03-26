#!/bin/bash

for i in 0 1 2 3 4
do
# print debug info
echo "<======================================== run $i"

../bin/tum_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_desk.yaml \
    ../cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
    /home/hnu/SamsungSSD2T/datasets/TUM/rgbd_dataset_freiburg1_desk \
    ../cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_desk.txt \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/TUM/RGBD/tum_rgbd_$i/rgbd_dataset_freiburg1_desk \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/tum_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg2_xyz.yaml \
    ../cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
    /home/hnu/SamsungSSD2T/datasets/TUM/rgbd_dataset_freiburg2_xyz \
    ../cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg2_xyz.txt \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/TUM/RGBD/tum_rgbd_$i/rgbd_dataset_freiburg2_xyz \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/tum_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
    ../cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
    /home/hnu/SamsungSSD2T/datasets/TUM/rgbd_dataset_freiburg3_long_office_household \
    ../cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/TUM/RGBD/tum_rgbd_$i/rgbd_dataset_freiburg3_long_office_household \
    no_viewer

echo "run $i ========================================>"
done
