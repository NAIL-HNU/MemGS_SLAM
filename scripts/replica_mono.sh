#!/bin/bash

for i in 0 1 2 3 4
do
# print debug info
echo "<======================================== run $i"

../bin/replica_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/Replica/office0.yaml \
    ../cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/hnu/Downloads/datasets/3dgs/Replica/office0 \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/Replica/Mono/replica_mono_$i/office0 \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/replica_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/Replica/office1.yaml \
    ../cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/hnu/Downloads/datasets/3dgs/Replica/office1 \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/Replica/Mono/replica_mono_$i/office1 \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/replica_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/Replica/office2.yaml \
    ../cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/hnu/Downloads/datasets/3dgs/Replica/office2 \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/Replica/Mono/replica_mono_$i/office2 \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/replica_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/Replica/office3.yaml \
    ../cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/hnu/Downloads/datasets/3dgs/Replica/office3 \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/Replica/Mono/replica_mono_$i/office3 \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/replica_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/Replica/office4.yaml \
    ../cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/hnu/Downloads/datasets/3dgs/Replica/office4 \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/Replica/Mono/replica_mono_$i/office4 \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/replica_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/Replica/room0.yaml \
    ../cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/hnu/Downloads/datasets/3dgs/Replica/room0 \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/Replica/Mono/replica_mono_$i/room0 \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/replica_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/Replica/room1.yaml \
    ../cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/hnu/Downloads/datasets/3dgs/Replica/room1 \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/Replica/Mono/replica_mono_$i/room1 \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/replica_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/Replica/room2.yaml \
    ../cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/hnu/Downloads/datasets/3dgs/Replica/room2 \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/Replica/Mono/replica_mono_$i/room2 \
    no_viewer

echo "run $i ========================================>"
done
