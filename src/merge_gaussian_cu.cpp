#include "../include/merge_gaussian_cu.h"
#include <ATen/core/TensorBody.h>
#include <driver_types.h>

void CallMergeGaussianInCUDA::MergeGaussian::calculateMDInCUDA(torch::Tensor& clipped_xyz, torch::Tensor& clipped_cov3D_inv, 
                                                                torch::Tensor& merged_idxs_tmp, torch::Tensor& merged_MD ) {

        int size = merged_idxs_tmp.size(0);  

        callCalculateMDInCUDA(size, clipped_xyz.contiguous().data_ptr<float>(), clipped_cov3D_inv.contiguous().data_ptr<float>(), 
                                merged_idxs_tmp.contiguous().data_ptr<int>(), merged_MD.contiguous().data_ptr<float>());
        CHECK_CUDA(, true);

}

std::tuple<torch::Tensor, torch::Tensor> 
CallMergeGaussianInCUDA::MergeGaussian::optimizeRotationAndScaleLBFGS(torch::Tensor& initQuat, torch::Tensor& initScale, 
                                                                      torch::Tensor& sigmaA, torch::Tensor& sigmaB,
                                                                      double lr , int max_iter ,
                                                                      double abs_tol , double rel_tol , double grad_tol ) {
    initScale = torch::clamp_min(initScale, 1e-6);  

    torch::Tensor optQuat   = initQuat.clone().detach().requires_grad_(true);           
    torch::Tensor optScale  = initScale.clone().detach().requires_grad_(true);    

    torch::optim::LBFGSOptions lbfgs_opts(lr);
    lbfgs_opts.max_iter(1);  
    lbfgs_opts.line_search_fn("strong_wolfe");  
    lbfgs_opts.tolerance_grad(1e-8);       
    lbfgs_opts.tolerance_change(1e-10);    
    lbfgs_opts.history_size(20);           
    torch::optim::LBFGS optimizer({optQuat, optScale}, lbfgs_opts);

    double prev_loss = std::numeric_limits<double>::max();
    for (int iter = 0; iter < max_iter; ++iter) {

        auto closure = [&]() -> torch::Tensor {
            optimizer.zero_grad();
            {
                torch::NoGradGuard no_grad;  
                optQuat /= optQuat.norm(2, 1, true).clamp_min(1e-12);  
                optScale.clamp_min_(1e-6).clamp_max_(1e6);  
            }
            torch::Tensor loss  = objective(optQuat, optScale, sigmaA, sigmaB);                   

            loss.backward();
            return loss;
        };

        auto loss_val = optimizer.step(closure);
        double curr_loss = loss_val.item<double>();

        double abs_loss = std::abs(prev_loss - curr_loss);
        double rel_loss = abs_loss / (std::abs(prev_loss) + 1e-12);  
        double gnorm = 0.0f;
        {
            torch::NoGradGuard no_grad;  
            for (auto *p : {&optQuat, &optScale}) {
                if (p->grad().defined()) {
                    double n = p->grad().norm().item<double>();
                    gnorm += n * n;  
                }
            }
            gnorm = std::sqrt(gnorm); 
        }
        
        if (abs_loss < abs_tol || rel_loss < rel_tol) { break; }
        prev_loss = curr_loss;
    }
    torch::NoGradGuard no_grad;        
    optQuat  = optQuat / optQuat.norm(2, 1, true).clamp_min(1e-12);
    optScale = optScale.clamp_min(1e-6).clamp_max(1e6);  

    return std::make_tuple(optQuat.detach(), optScale.detach());
}

torch::Tensor CallMergeGaussianInCUDA::MergeGaussian::objective(const torch::Tensor& optQuat, const torch::Tensor& optScale, 
                                                                const torch::Tensor& sigmaA, const torch::Tensor& sigmaB) {
    // 检查scale是否为正数
    if (!optScale.isfinite().all().item<bool>()) {
        // std::cout << "\n待优化尺度参数包含无效值(NaN或Inf): \n" << optScale << std::endl;
        // std::cout << "\n待优化旋转参数包含无效值: \n" << optQuat << std::endl;

        // 抓取原始数据进行分析
        torch::Tensor non_mask_scale = optScale.isnan().any(/*dim=*/1) | optScale.isinf().any(/*dim=*/1);
        torch::Tensor non_mask_scale_idx = torch::nonzero(non_mask_scale).squeeze();
        // 输出对应的sigmaA和sigmaB
        torch::Tensor non_mask_sigmaA = sigmaA.index({non_mask_scale_idx, torch::indexing::Slice(), torch::indexing::Slice()});
        torch::Tensor non_mask_sigmaB = sigmaB.index({non_mask_scale_idx, torch::indexing::Slice(), torch::indexing::Slice()});
        std::cout << ClrUtils::RED << "\n待优化尺度参数包含无效值(NaN或Inf)的索引: \n" << non_mask_scale_idx << ClrUtils::RESET << std::endl;
        std::cout << "\n对应的sigmaA: \n" << non_mask_sigmaA << std::endl;
        std::cout << "\n对应的sigmaB: \n" << non_mask_sigmaB << std::endl;

    }

    torch::Tensor RotationMat = build_rotation(optQuat);       
    torch::Tensor DiagonalMat = torch::diag_embed(optScale.pow(2));    
    torch::Tensor SigmaC      = RotationMat.matmul(DiagonalMat).matmul(RotationMat.transpose(-2, -1)); 

    SigmaC = 0.5 * (SigmaC + SigmaC.transpose(-2, -1));
    SigmaC = SigmaC + 1e-6 * torch::eye(3, SigmaC.options()).unsqueeze(0);

    torch::Tensor sqrtSigmaC    = matrix_square_root(SigmaC);   
    torch::Tensor C2A           = wasserstein_distance(sqrtSigmaC, SigmaC, sigmaA);     
    torch::Tensor C2B           = wasserstein_distance(sqrtSigmaC, SigmaC, sigmaB);    

    return (C2A + C2B).mean();  
}

torch::Tensor CallMergeGaussianInCUDA::MergeGaussian::matrix_square_root(const torch::Tensor& matrix, double eps_scale) {
    torch::Tensor regularized_matrix = matrix.to(TorchUtils::kFloat64CUDA) + eps_scale * torch::eye(matrix.size(-1), TorchUtils::kFloat64CUDA);

    try {
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> svd_result = torch::linalg::svd(regularized_matrix, false /* full_matrices */ , "gesvd" );
        torch::Tensor U    = std::get<0>(svd_result);
        torch::Tensor S    = std::get<1>(svd_result);

        torch::Tensor min_singular_value    = torch::amin(S, /*dim=*/1);
        torch::Tensor Vh             = std::get<2>(svd_result);
        torch::Tensor S_sqrt         = torch::sqrt(torch::clamp(S, eps_scale));            
        torch::Tensor S_sqrt_diag    = torch::diag_embed(S_sqrt);                          

        return (U.matmul(S_sqrt_diag).matmul(Vh)).to(matrix.options());                    
        
    } catch (const c10::Error& e) {
        std::stringstream ss;
        throw std::runtime_error(ss.str());
    }
}

torch::Tensor CallMergeGaussianInCUDA::MergeGaussian::wasserstein_distance(const torch::Tensor& sqrt_SigmaA, const torch::Tensor& sigmaA, const torch::Tensor& sigmaB) {
    if (!sigmaA.is_contiguous() || !sigmaB.is_contiguous() || !sqrt_SigmaA.is_contiguous()) TORCH_WARN("In wasserstein_distance function, input tensors are not contiguous.");

    torch::Tensor cross_sqrt = matrix_square_root(sqrt_SigmaA.matmul(sigmaB).matmul(sqrt_SigmaA));           
    if (torch::any(torch::isnan(cross_sqrt)).item<bool>()) TORCH_WARN("In wasserstein_distance function, cross_sqrt contains NaN values.");

    torch::Tensor wasserstein_matrix  = sigmaA.add(sigmaB).sub(cross_sqrt.mul(2));                           
    return wasserstein_matrix.diagonal(/* offset */ 0, /* dim1 */ -2, /* dim2 */ -1).sum(-1);              
}

torch::Tensor CallMergeGaussianInCUDA::MergeGaussian::slerp(const torch::Tensor& q1, const torch::Tensor& q2, float t ) {

    torch::Tensor dot           = torch::sum(q1 * q2, -1);                      
    torch::Tensor q2_adj        = torch::where(dot.unsqueeze(-1) < 0, -q2, q2);

    dot         = torch::sum(q1 * q2_adj, -1);                                  
    dot         = dot.clamp(-1.0f, 1.0f);

    torch::Tensor theta_0        = torch::acos(dot);                             
    torch::Tensor sin_theta_0    = torch::sin(theta_0);                          
    torch::Tensor is_degenerate  = sin_theta_0.abs() < 1e-6;

    torch::Tensor theta          = theta_0 * t;
    torch::Tensor sin_theta      = torch::sin(theta);
    torch::Tensor scale1         = torch::sin((1.0f - t) * theta_0) / sin_theta_0;
    torch::Tensor scale2         = sin_theta / sin_theta_0;
    
    scale1      = scale1.unsqueeze(-1);
    scale2      = scale2.unsqueeze(-1);

    torch::Tensor result         = scale1 * q1 + scale2 * q2_adj;

    auto nn_options              = torch::nn::functional::NormalizeFuncOptions().dim(-1);
    torch::Tensor lerp_result    = torch::nn::functional::normalize((1 - t) * q1 + t * q2_adj, nn_options);
    result      = torch::where(is_degenerate.unsqueeze(-1), lerp_result, result);

    return torch::nn::functional::normalize(result, nn_options);
}

torch::Tensor CallMergeGaussianInCUDA::MergeGaussian::build_rotation(const torch::Tensor& r, bool need_normalized ) {  // r (w, x, y, z)
    torch::Tensor norm = r.norm(/* p */2, /* dim */1, /* keepdim */true);        
    torch::Tensor q;
    if (need_normalized) {   
        q = r / norm;       
    } else {
        torch::Tensor ones     = torch::ones_like(norm, norm.options());         
        bool is_unit           = torch::allclose(norm, ones, 1e-3, 1e-3);

        if (!is_unit) {     
            torch::Tensor diff        = (norm - ones).abs();                     
            torch::Tensor diff_flat   = diff.squeeze();                         
            int count                 = std::min(static_cast<int>(diff_flat.size(0)), 10);
            torch::Tensor diff_top10  = diff_flat.index({torch::indexing::Slice(0, count)});
        }
        q = r;
    }

    torch::Tensor w = q.index({torch::indexing::Slice(), 0});
    torch::Tensor x = q.index({torch::indexing::Slice(), 1});
    torch::Tensor y = q.index({torch::indexing::Slice(), 2});
    torch::Tensor z = q.index({torch::indexing::Slice(), 3});

    torch::Tensor R = torch::empty({q.size(0), 3, 3}, q.options());              
    R.index_put_({torch::indexing::Slice(), 0, 0}, 1 - 2 * (y * y + z * z));
    R.index_put_({torch::indexing::Slice(), 0, 1}, 2 * (x * y - w * z));
    R.index_put_({torch::indexing::Slice(), 0, 2}, 2 * (x * z + w * y));
    R.index_put_({torch::indexing::Slice(), 1, 0}, 2 * (x * y + w * z));
    R.index_put_({torch::indexing::Slice(), 1, 1}, 1 - 2 * (x * x + z * z));
    R.index_put_({torch::indexing::Slice(), 1, 2}, 2 * (y * z - w * x));
    R.index_put_({torch::indexing::Slice(), 2, 0}, 2 * (x * z - w * y));
    R.index_put_({torch::indexing::Slice(), 2, 1}, 2 * (y * z + w * x));
    R.index_put_({torch::indexing::Slice(), 2, 2}, 1 - 2 * (x * x + y * y));

    return R;
}