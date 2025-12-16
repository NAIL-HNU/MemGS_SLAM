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

#include <tuple>

#include <torch/torch.h>

#include "rasterize_points.h"
#include "gaussian_model.h"

struct GaussianRasterizationSettings
{
    GaussianRasterizationSettings(
        int image_height,
        int image_width,
        float tanfovx,
        float tanfovy,
        torch::Tensor& bg,
        float scale_modifier,
        torch::Tensor& viewmatrix,
        torch::Tensor& projmatrix,
        torch::Tensor& projmatrix_raw,
        int sh_degree,
        torch::Tensor& campos,
        bool prefiltered)
        : image_height_(image_height), image_width_(image_width), tanfovx_(tanfovx), tanfovy_(tanfovy),
          bg_(bg), scale_modifier_(scale_modifier), viewmatrix_(viewmatrix), projmatrix_(projmatrix),
          sh_degree_(sh_degree), campos_(campos), prefiltered_(prefiltered), projmatrix_raw_(projmatrix_raw)
    {}

    int image_height_;
    int image_width_;
    float tanfovx_;
    float tanfovy_;
    torch::Tensor bg_;
    float scale_modifier_;
    torch::Tensor viewmatrix_;  /* viewpoint_camera: world_view_transform */
    torch::Tensor projmatrix_;  /* viewpoint_camera: full_proj_transform */
    torch::Tensor projmatrix_raw_;  /* viewpoint_camera: projection_matrix */
    int sh_degree_;
    torch::Tensor campos_;
    bool prefiltered_;
    bool debug_ = false;
};

class GaussianRasterizerFunction : public torch::autograd::Function<GaussianRasterizerFunction>
{
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor means3D,
        torch::Tensor means2D,
        torch::Tensor sh,
        torch::Tensor colors_precomp,
        torch::Tensor opacities,
        torch::Tensor scales,
        torch::Tensor rotations,
        torch::Tensor cov3Ds_precomp,
        GaussianRasterizationSettings raster_settings);

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_out_color);
};

inline torch::autograd::tensor_list rasterizeGaussians(
    torch::Tensor& means3D,
    torch::Tensor& means2D,
    torch::Tensor& sh,
    torch::Tensor& colors_precomp,
    torch::Tensor& opacities,
    torch::Tensor& scales,
    torch::Tensor& rotations,
    torch::Tensor& cov3Ds_precomp,
    GaussianRasterizationSettings& raster_settings)
{
    // 调用GaussianRasterizerFunction::forward函数
    return GaussianRasterizerFunction::apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings
    );
}

class GaussianRasterizer : public torch::nn::Module
{
public:
    GaussianRasterizer(GaussianRasterizationSettings& raster_settings)
        : raster_settings_(raster_settings)
    {}

    torch::Tensor markVisibleGaussians(torch::Tensor& positions);

    // 对外在render中被调用 forward内部实现调用rasterizeGaussians
    std::tuple<torch::Tensor/* rgb */, torch::Tensor/* radii */, torch::Tensor/* depth */, torch::Tensor/* opacity */, torch::Tensor/* n_touched */> forward(
        torch::Tensor means3D,
        torch::Tensor means2D,
        torch::Tensor opacities,
        bool has_shs,
        bool has_colors_precomp,
        bool has_scales,
        bool has_rotations,
        bool has_cov3D_precomp,
        torch::Tensor shs,
        torch::Tensor colors_precomp,
        torch::Tensor scales,
        torch::Tensor rotations,
        torch::Tensor cov3D_precomp);

public:
    GaussianRasterizationSettings raster_settings_;
};
