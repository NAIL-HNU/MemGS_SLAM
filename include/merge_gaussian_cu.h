#pragma once
#include "torch/serialize/input-archive.h"

// cuda
#include <ATen/core/TensorBody.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <tuple>
#include <torch/optim/optimizer.h>
#include <torch/optim/lbfgs.h>
#include <torch/nn/utils/clip_grad.h>

#include "general_utils.h"

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
    auto ret = cudaDeviceSynchronize(); \
    if (ret != cudaSuccess) { \
        printf("\n[CUDA ERROR] in %s\nLine %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); \
        throw std::runtime_error(cudaGetErrorString(ret)); \
    } \
}
namespace CallMergeGaussianInCUDA {
        struct ClrUtils {
        inline static const std::string BLACK = "\033[1;30m";
        inline static const std::string RED = "\033[1;31m";
        inline static const std::string GREEN = "\033[1;32m";
        inline static const std::string YELLOW = "\033[1;33m";
        inline static const std::string BLUE = "\033[1;34m";
        inline static const std::string MAGENTA = "\033[1;35m";
        inline static const std::string CYAN = "\033[1;36m";
        inline static const std::string WHITE = "\033[1;37m";
        inline static const std::string RESET = "\033[0m";
    };

    struct TorchUtils {
        // CUDA Tensor Options
        inline static const auto kFloat32CUDA = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        inline static const auto kFloat32CPU = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

        inline static const auto kFloat64CUDA = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
        inline static const auto kFloat64CPU = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

        inline static const auto kInt32CUDA = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        inline static const auto kInt32CPU = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

        inline static const auto kBoolCUDA = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
        inline static const auto kBoolCPU = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
    };
    class MergeGaussian {
        public:
            static void calculateMDInCUDA(torch::Tensor& clipped_xyz, torch::Tensor& clipped_cov3D_inv, torch::Tensor& merged_idxs_tmp, torch::Tensor& merged_MD);
            // ready for Kernel
            static void callCalculateMDInCUDA(int size, float* clipped_xyz, float* clipped_cov3D_inv, int* merged_idxs_tmp, float* merged_MD);


            static void MergeGaussiansInCUDA(torch::Tensor& clipped_xyz, torch::Tensor& clipped_opacity, 
                                            torch::Tensor& merged_SigmaA_inv, torch::Tensor& merged_SigmaB_inv, torch::Tensor& merged_cov3D_inv,
                                            torch::Tensor& merged_xyz, torch::Tensor& merged_opacity, torch::Tensor& merged_idxs);
            static std::tuple<torch::Tensor, torch::Tensor> 
            optimizeRotationAndScaleLBFGS(torch::Tensor& initQuat, torch::Tensor& initScale, 
                                          torch::Tensor& sigmaA, torch::Tensor& sigmaB,
                                          double lr = 1e-3, int max_iter = 15,
                                          double abs_tol = 1e-6, double rel_tol = 1e-6, double grad_tol = 1e-12);
                                                                                                                          
            static torch::Tensor objective(const torch::Tensor& optQuat, const torch::Tensor& optScale, 
                                            const torch::Tensor& sigmaA, const torch::Tensor& sigmaB);
            static torch::Tensor matrix_square_root(const torch::Tensor& matrix, double eps_scale = 1e-12);
            static torch::Tensor wasserstein_distance(const torch::Tensor& sqrt_SigmaA, const torch::Tensor& sigmaA, const torch::Tensor& sigmaB);
            static torch::Tensor slerp(const torch::Tensor& q1, const torch::Tensor& q2, float t = 0.5);
            static torch::Tensor build_rotation(const torch::Tensor& r, bool need_normalized = true);
    };
};