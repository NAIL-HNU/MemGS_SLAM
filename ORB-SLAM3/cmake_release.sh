clear && /usr/bin/cmake -B build -DCMAKE_C_COMPILER=$(which gcc) -DCMAKE_CXX_COMPILER=$(which g++) \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
-DCMAKE_BUILD_TYPE=Release . && /usr/bin/cmake --build build --parallel $(nproc --all)
# -DPYTHON_INCLUDE_DIRS=/home/robot/anaconda3/envs/photo_slam_cu118_py20/include/python3.10 \
# -DPYTHON_LIBRARIES=/home/robot/anaconda3/envs/photo_slam_cu118_py20/lib/libpython3.10.so \
# -DPYTHON_EXECUTABLE=/home/robot/anaconda3/envs/photo_slam_cu118_py20/bin/python3.10 \
# -DPYTHON_NUMPY_INCLUDE_DIR=/home/robot/anaconda3/envs/photo_slam_cu118_py20/lib/python3.10/site-packages/numpy/core/include \
# -DASAN_ENABLED=ON \ 
