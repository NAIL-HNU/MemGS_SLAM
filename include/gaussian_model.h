/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM.
 */

#pragma once

#include <memory>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

#include "third_party/simple-knn/spatial.h"
#include "third_party/tinyply/tinyply.h"
#include "types.h"
#include "point3d.h"
#include "operate_points.h"
#include "general_utils.h"
#include "sh_utils.h"
#include "tensor_utils.h"
#include "gaussian_parameters.h"

#include "merge_gaussian_cu.h"

#define GAUSSIAN_MODEL_TENSORS_TO_VEC                        \
    this->Tensor_vec_xyz_ = {this->xyz_};                    \
    this->Tensor_vec_feature_dc_ = {this->features_dc_};     \
    this->Tensor_vec_feature_rest_ = {this->features_rest_}; \
    this->Tensor_vec_opacity_ = {this->opacity_};            \
    this->Tensor_vec_scaling_ = {this->scaling_};            \
    this->Tensor_vec_rotation_ = {this->rotation_};

#define GAUSSIAN_MODEL_INIT_TENSORS(device_type)                                             \
    this->xyz_ = torch::empty(0, torch::TensorOptions().device(device_type));                \
    this->features_dc_ = torch::empty(0, torch::TensorOptions().device(device_type));        \
    this->features_rest_ = torch::empty(0, torch::TensorOptions().device(device_type));      \
    this->scaling_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->rotation_ = torch::empty(0, torch::TensorOptions().device(device_type));           \
    this->opacity_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->max_radii2D_ = torch::empty(0, torch::TensorOptions().device(device_type));        \
    this->xyz_gradient_accum_ = torch::empty(0, torch::TensorOptions().device(device_type)); \
    this->denom_ = torch::empty(0, torch::TensorOptions().device(device_type));              \
    this->unique_kfID = torch::empty(0, torch::TensorOptions().device(device_type));         \
    GAUSSIAN_MODEL_TENSORS_TO_VEC

// #include <Eigen/Dense>
#include <Eigen/Core>
struct hash_vec {
    std::size_t operator()(const Eigen::Vector3i& key) const noexcept {
        // 按元素读 int，再转成 uint32_t，避免直接 reinterpret_cast 带来的未定义行为
        const int* data = key.data();

        // 三个常用的“voxel 哈希”质数（你给的那三个）
        static constexpr std::uint32_t primes[3] = {
            73856093u,
            19349669u,
            83492791u
        };

        std::size_t h = 0;
        for (std::size_t i = 0; i < 3; ++i) {
            std::uint32_t v = static_cast<std::uint32_t>(data[i]);
            h ^= static_cast<std::size_t>(v) * primes[i];
        }

        // 对于 std::unordered_map，直接返回 h 即可，没必要再 & ((1<<Log2N)-1)
        return h;
    }
};

struct key_equal_vec {
    bool operator()(const Eigen::Vector3i& a, const Eigen::Vector3i& b) const noexcept {
        return a == b;
    }
};

class GaussianModel
{
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

public:
    GaussianModel(const int sh_degree);
    GaussianModel(const GaussianModelParams& model_params);

    torch::Tensor getScalingActivation();
    torch::Tensor getRotationActivation();
    torch::Tensor getXYZ();
    torch::Tensor getFeatures();
    torch::Tensor getOpacityActivation();
    torch::Tensor getCovarianceActivation(int scaling_modifier = 1);

    void oneUpShDegree();
    void setShDegree(const int sh);

    void createFromPcd(
        std::map<point3D_id_t, Point3D> pcd,
        const float spatial_lr_scale,
        const int kf_id = 0);

    void increasePcd(std::vector<float> points, std::vector<float> colors, const int iteration, const int kf_id = 0);
    void increasePcd(torch::Tensor& new_point_cloud, torch::Tensor& new_colors, const int iteration, const int kf_id = 0);

    void applyScaledTransformation(
        const float s = 1.0,
        const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));
    void scaledTransformationPostfix(
        torch::Tensor& new_xyz,
        torch::Tensor& new_scaling);

    void scaledTransformVisiblePointsOfKeyframe(
        torch::Tensor& point_not_transformed_flags,
        torch::Tensor& diff_pose,
        torch::Tensor& kf_world_view_transform,
        torch::Tensor& kf_full_proj_transform,
        const int kf_creation_iter,
        const int stable_num_iter_existence,
        int& num_transformed,
        const float scale = 1.0f);

    void trainingSetup(const GaussianOptimizationParams& training_args);
    float updateLearningRate(int step);
    void setPositionLearningRate(float position_lr);
    void setFeatureLearningRate(float feature_lr);
    void setOpacityLearningRate(float opacity_lr);
    void setScalingLearningRate(float scaling_lr);
    void setRotationLearningRate(float rot_lr);

    void resetOpacity();
    torch::Tensor replaceTensorToOptimizer(torch::Tensor& t, int tensor_idx);

    void prunePoints(torch::Tensor& mask);

    void densificationPostfix(
        torch::Tensor& new_xyz,
        torch::Tensor& new_features_dc,
        torch::Tensor& new_features_rest,
        torch::Tensor& new_opacities,
        torch::Tensor& new_scaling,
        torch::Tensor& new_rotation,
        torch::Tensor& new_exist_since_iter,
        torch::Tensor& new_kf_ids);

    void densifyAndSplit(
        torch::Tensor& grads,
        float grad_threshold,
        float scene_extent,
        int N = 2);

    void densifyAndClone(
        torch::Tensor& grads,
        float grad_threshold,
        float scene_extent);

    void densifyAndPrune(
        float max_grad,
        float min_opacity,
        float extent,
        int max_screen_size);

    void addDensificationStats(
        torch::Tensor& viewspace_point_tensor,
        torch::Tensor& update_filter);

// void increasePointsIterationsOfExistence(const int i = 1);

    void loadPly(std::filesystem::path ply_path);
    void savePly(std::filesystem::path result_path);
    void saveSparsePointsPly(std::filesystem::path result_path);

    float percentDense();
    void setPercentDense(const float percent_dense);

protected:
    float exponLrFunc(int step);

public:
    torch::DeviceType device_type_;

    int active_sh_degree_;
    int max_sh_degree_;

    torch::Tensor xyz_;
    torch::Tensor features_dc_;
    torch::Tensor features_rest_;
    torch::Tensor scaling_;
    torch::Tensor rotation_;
    torch::Tensor opacity_;
    torch::Tensor max_radii2D_;
    torch::Tensor xyz_gradient_accum_;
    torch::Tensor denom_;
    torch::Tensor exist_since_iter_;
    torch::Tensor unique_kfID;

    std::vector<torch::Tensor> Tensor_vec_xyz_,
                               Tensor_vec_feature_dc_,
                               Tensor_vec_feature_rest_,
                               Tensor_vec_opacity_,
                               Tensor_vec_scaling_ ,
                               Tensor_vec_rotation_;

    std::shared_ptr<torch::optim::Adam> optimizer_;
    float percent_dense_;
    float spatial_lr_scale_;

    torch::Tensor sparse_points_xyz_;
    torch::Tensor sparse_points_color_;

protected:
    float lr_init_;
    float lr_final_;
    int lr_delay_steps_;
    float lr_delay_mult_;
    int max_steps_;

    std::mutex mutex_settings_;

public:
    inline Eigen::Vector3i Pos2Grid(float x, float y, float z) noexcept {
        Eigen::Vector3i index;
        static float inv_resolution = 1.0f / merge_params.resolution;
        index[0] = static_cast<int>(std::roundf(x * inv_resolution));
        index[1] = static_cast<int>(std::roundf(y * inv_resolution));
        index[2] = static_cast<int>(std::roundf(z * inv_resolution));
        return index;
    }
    void insertPointsToVoxel(float x, float y, float z, std::size_t point_idx);
    void traverseVoxel();

    void mergeGaussiansByMDInCUDA(torch::Tensor& clipped_mask, bool merge_still_reserved = true);
    void calculateMDInCUDA(torch::Tensor& merged_idxs_tmp, torch::Tensor& clipped_mask, torch::Tensor& merged_MD);
    void MergeGaussiansInCUDA(torch::Tensor& reserved_idxs, torch::Tensor& merged_idxs, torch::Tensor& clipped_mask);

    torch::Tensor getActualCovarianceActivation(int scaling_modifier = 1);

    struct MergeParams {
        bool verbose = true;
        float resolution = 0.01f;
        bool enable_LBFGS_opt = true;
    } merge_params;
    int vox_pts_num_size = 500, merged_pts_size = 1000, reserved_pts_num_size = 1000;

    std::size_t over_pair_cnt = 0;                                              
    std::size_t pts_num = 0;                                                   
    std::vector<int> vox_pts_num;                                              
    std::vector<int> merged_pts_idxs;                                         
    std::vector<int> reserved_pt_idxs;                                        
    static constexpr int SINGLE_POINT_VOXEL = 0;                             
    std::unordered_map<Eigen::Vector3i, std::vector<std::size_t>, hash_vec, key_equal_vec> voxel_map; 

};
