#include <vector>
#include <cstdio>

const unsigned short grain = 1;

template <typename scalar_t>
__global__ void sage_conv_cuda_forward_kernel(
        const int n_vertex,
        const int fsize,
        scalar_t * features,
        int * col_starts,
        int * rows,
        scalar_t * result) {

    int des_v;

    des_v = blockIdx.x * blockDim.y + threadIdx.y;

    if (des_v < n_vertex)
    {
        scalar_t * des_p = result + (int64_t)des_v * 2 * fsize;
        scalar_t * src_p = features + des_v * fsize;
        int s_pos = col_starts[des_v];
        int e_pos = col_starts[des_v+1];
        scalar_t deg = 1.0 / (e_pos - s_pos);
        scalar_t ret;
        for (int k = threadIdx.x; k < fsize; k += blockDim.x) {
            des_p[k] = src_p[k];
            ret = 0.0;
            for (int i = s_pos; i < e_pos; ++i)
            {
                // src_p = features + src_v * fsize;
                // ret += src_p[k];
                ret += __ldg(features + rows[i] * fsize + k);
            }

            des_p += fsize;
            des_p[k] = ret * deg;
        }

    }
}

std::vector<torch::Tensor> sage_conv_cuda_forward(
        torch::Tensor features,
        torch::Tensor col_starts,
        torch::Tensor rows) {

    auto feature_size = features.size(1);

    auto result = torch::empty({features.size(0), 2*feature_size}, features.options());

    const int w = 8;
    const int xaxis = (feature_size <= 16 ? 16 : 32);
    const dim3 threads(xaxis, w);
    const int blocks = (features.size(0) + w - 1) / w;
    // const int blocks = 640;

    // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "graphsage_forward_cuda", ([&] {
        sage_conv_cuda_forward_kernel<<<blocks, threads>>>(
            features.size(0),
            feature_size,
            features.data_ptr<scalar_t>(),
            col_starts.data_ptr<int>(),
            rows.data_ptr<int>(),
            result.data_ptr<scalar_t>()
        );
    }));

    return {result};
}

template <typename scalar_t>
__global__ void sage_conv_cuda_backward_kernel(
        int n_vertex,
        scalar_t * grad,
        scalar_t * indegs,
        int * row_starts,
        int * cols,
        scalar_t * result) {

    int src_v = blockIdx.x * blockDim.y + threadIdx.y;
    if (src_v < n_vertex)
    {
        scalar_t * src_p = result + src_v * blockDim.x;
        scalar_t * des_p = grad + src_v * 2 * blockDim.x;
        int s_pos = row_starts[src_v];
        int e_pos = row_starts[src_v+1];
        scalar_t ret = des_p[threadIdx.x];
        for (int i = s_pos; i < e_pos; ++i)
        {
            int des_v = cols[i];
            des_p = grad + des_v * 2 * blockDim.x + blockDim.x;
            ret += des_p[threadIdx.x] / indegs[des_v];
        }

        src_p[threadIdx.x] = ret;
    }
}

std::vector<torch::Tensor> sage_conv_cuda_backward(
        torch::Tensor features,
        torch::Tensor grad,
        torch::Tensor indegs,
        torch::Tensor row_starts,
        torch::Tensor cols) {

    auto feature_size = features.size(1);

    auto result = torch::empty_like(features);

    const int w = 8;
    const dim3 threads(features.size(1), w);
    const int blocks = (features.size(0) + w - 1) / w;

    // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "graphsage_backward_cuda", ([&] {
        sage_conv_cuda_backward_kernel<<<blocks, threads>>>(
            features.size(0),
            grad.data_ptr<scalar_t>(),
            indegs.data_ptr<scalar_t>(),
            row_starts.data_ptr<int>(),
            cols.data_ptr<int>(),
            result.data_ptr<scalar_t>()
        );
    }));

    return {result};
}

