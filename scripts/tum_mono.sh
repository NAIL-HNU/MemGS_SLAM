#!/bin/bash

for i in 0 1 2 3 4
do
# print debug info
echo "<======================================== run $i"

../bin/tum_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg1_desk.yaml \
    ../cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
    /home/hnu/SamsungSSD2T/datasets/TUM/rgbd_dataset_freiburg1_desk \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/TUM/Mono/tum_mono_$i/rgbd_dataset_freiburg1_desk \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/tum_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg2_xyz.yaml \
    ../cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
    /home/hnu/SamsungSSD2T/datasets/TUM/rgbd_dataset_freiburg2_xyz \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/TUM/Mono/tum_mono_$i/rgbd_dataset_freiburg2_xyz \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/tum_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg3_long_office_household.yaml \
    ../cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
    /home/hnu/SamsungSSD2T/datasets/TUM/rgbd_dataset_freiburg3_long_office_household \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/TUM/Mono/tum_mono_$i/rgbd_dataset_freiburg3_long_office_household \
    no_viewer

echo "run $i ========================================>"
done
