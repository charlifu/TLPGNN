#include <vector>
#include <cstdio>

const unsigned short grain = 1;

template <typename scalar_t>
__global__ void gin_conv_cuda_forward_kernel(
        const int n_vertex,
        const int fsize,
        scalar_t * features,
        scalar_t * eps,
        int * col_starts,
        int * rows,
        scalar_t * result) {
    int des_v;

    des_v = blockIdx.x * blockDim.y + threadIdx.y;

    // while (des_v < n_vertex)
    if (des_v < n_vertex)
    {
        scalar_t ret; 
        int s_pos = col_starts[des_v];
        int e_pos = col_starts[des_v+1];
        scalar_t * des_p = result + des_v * fsize;
        for (int k = threadIdx.x; k < fsize; k += blockDim.x) {
            ret = (scalar_t(1.0) + *eps) * features[des_v * fsize + k];
            for (int i = s_pos; i < e_pos; ++i)
            {
                ret += __ldg(features + rows[i] * fsize + k);
            }
            des_p[k] = ret;
        }
    }
}

std::vector<torch::Tensor> gin_conv_cuda_forward(
        torch::Tensor features,
        torch::Tensor eps,
        torch::Tensor col_starts,
        torch::Tensor rows) {
    auto result = torch::empty_like(features);

    const int w = 4;
    const int feature_size = features.size(1);

    const int xaxis = (feature_size <= 16 ? 16 : 32);
    const dim3 threads(xaxis, w);
    const int blocks = (features.size(0) + w - 1) / w;

    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "aggregate_forward_cuda", ([&] 
    {
        gin_conv_cuda_forward_kernel<<<blocks, threads>>>(
            features.size(0),
            feature_size,
            features.data_ptr<scalar_t>(),
            eps.data_ptr<scalar_t>(),
            col_starts.data_ptr<int>(),
            rows.data_ptr<int>(),
            result.data_ptr<scalar_t>()
        );
    }));

    return {result};
}

template <typename scalar_t>
__global__ void gin_conv_cuda_backward_kernel(
        int n_vertex,
        scalar_t * grad,
        scalar_t * eps,
        int * row_starts,
        int * cols,
        scalar_t * result) {

    int src_v = blockIdx.x * blockDim.y + threadIdx.y;

    if (src_v < n_vertex)
    {
        scalar_t ret = (scalar_t(1.0) + *eps) * grad[src_v * blockDim.x + threadIdx.x];
        int s_pos = row_starts[src_v];
        int e_pos = row_starts[src_v+1];
        scalar_t * src_p = result + src_v * blockDim.x;
        for (int i = s_pos; i < e_pos; ++i)
        {
            int des_v = cols[i];
            scalar_t * des_p = grad + des_v * blockDim.x;
            ret += des_p[threadIdx.x];
        }

        src_p[threadIdx.x] = ret;
    }
}

std::vector<torch::Tensor> gin_conv_cuda_backward(
        torch::Tensor features,
        torch::Tensor grad,
        torch::Tensor eps,
        torch::Tensor row_starts,
        torch::Tensor cols) {
    auto result = torch::empty_like(features);

    const int w = 8;
    const int feature_size = features.size(1);

    const dim3 threads(features.size(1), w);
    const int blocks = (features.size(0) + w - 1) / w;

    // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "conv_backward_cuda", ([&] 
    {
        gin_conv_cuda_backward_kernel<<<blocks, threads>>>(
            features.size(0),
            grad.data_ptr<scalar_t>(),
            eps.data_ptr<scalar_t>(),
            row_starts.data_ptr<int>(),
            cols.data_ptr<int>(),
            result.data_ptr<scalar_t>()
        );
    }));

    return {result};
}
