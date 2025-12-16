#include "../include/merge_gaussian_cu.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <sys/types.h>
#include <cfloat>
#include <cstdint>

namespace MergeGaussiansInCUDA {
    namespace cg = cooperative_groups;
    __global__ void calculateMDInCUDA(int size, const float* clipped_xyz, const float* clipped_cov3D_inv, int* merged_idxs_tmp, float* merged_MD, float md_dist_th, bool debug=false);
};

namespace CallMergeGaussianInCUDA{
    void MergeGaussian::callCalculateMDInCUDA(int size, float* clipped_xyz, float* clipped_cov3D_inv, 
                                                int* merged_idxs_tmp, float* merged_MD ) {
        int block_size = 256;                                               
        int block_num = (size + block_size - 1) / block_size;               

        MergeGaussiansInCUDA::calculateMDInCUDA<<<block_num, block_size>>>(size, clipped_xyz, clipped_cov3D_inv, merged_idxs_tmp, merged_MD, general_utils::md_thr); 
    }

};

__global__ void MergeGaussiansInCUDA::calculateMDInCUDA(int size, const float* clipped_xyz, const float* clipped_cov3D_inv, 
                                                            int* merged_idxs_tmp, float* merged_MD, float md_dist_th, bool debug) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= size) return;
    int pair_idx= idx*2;
    int major_idx = merged_idxs_tmp[pair_idx];                          
    int second_idx = merged_idxs_tmp[pair_idx+1];                       

    float major_x = clipped_xyz[major_idx*3]; float major_y = clipped_xyz[major_idx*3+1]; float major_z = clipped_xyz[major_idx*3+2];
    float second_x = clipped_xyz[second_idx*3]; float second_y = clipped_xyz[second_idx*3+1]; float second_z = clipped_xyz[second_idx*3+2];

    float major_cov3D_inv[3][3] = {{clipped_cov3D_inv[major_idx*9], clipped_cov3D_inv[major_idx*9+1], clipped_cov3D_inv[major_idx*9+2]},
                            {clipped_cov3D_inv[major_idx*9+3], clipped_cov3D_inv[major_idx*9+4], clipped_cov3D_inv[major_idx*9+5]},
                            {clipped_cov3D_inv[major_idx*9+6], clipped_cov3D_inv[major_idx*9+7], clipped_cov3D_inv[major_idx*9+8]}};

    float mahalanobis_distance = 0.0f;
    float diff_x = major_x - second_x; float diff_y = major_y - second_y; float diff_z = major_z - second_z;
    float diff[3] = {diff_x, diff_y, diff_z};
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            mahalanobis_distance += diff[i] * major_cov3D_inv[i][j] * diff[j];
        }
    }

    if (mahalanobis_distance < md_dist_th) {
        merged_MD[idx] = mahalanobis_distance;                          
    } else {
        merged_MD[idx] = md_dist_th;                                     
    }
}
