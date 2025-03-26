#!/bin/zsh

for i in 0 1 2 3 4
do
echo "<======================================== run $i"

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/hnu/Downloads/datasets/3dgs/Replica/office0 \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/Replica/RGBD/replica_rgbd_$i/office0 \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office1.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/hnu/Downloads/datasets/3dgs/Replica/office1 \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/Replica/RGBD/replica_rgbd_$i/office1 \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office2.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/hnu/Downloads/datasets/3dgs/Replica/office2 \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/Replica/RGBD/replica_rgbd_$i/office2 \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office3.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/hnu/Downloads/datasets/3dgs/Replica/office3 \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/Replica/RGBD/replica_rgbd_$i/office3 \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office4.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/hnu/Downloads/datasets/3dgs/Replica/office4 \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/Replica/RGBD/replica_rgbd_$i/office4 \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/room0.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/hnu/Downloads/datasets/3dgs/Replica/room0 \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/Replica/RGBD/replica_rgbd_$i/room0 \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/room1.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/hnu/Downloads/datasets/3dgs/Replica/room1 \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/Replica/RGBD/replica_rgbd_$i/room1 \
    no_viewer

echo "**********************************************************************************"
echo "**********************************************************************************"

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/room2.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/hnu/Downloads/datasets/3dgs/Replica/room2 \
    /home/hnu/SamsungSSD2T/Results/Photo-SLAM/Replica/RGBD/replica_rgbd_$i/room2 \
    no_viewer

echo "run $i ========================================>"
done
