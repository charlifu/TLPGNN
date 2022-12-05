#include <vector>
#include <cstdio>
#include <typeinfo>

// const unsigned short grain = 4;
// const int grain = 48;

template <typename scalar_t>
__global__ void gcn_conv_cuda_forward_kernel(
        int * n_done,
        const int n_vertex,
        const int fsize,
        scalar_t * __restrict__ features,
        int * __restrict__ col_starts,
        int * __restrict__ rows,
        scalar_t * __restrict__ result) {
    
    int des_v;
    
    if(threadIdx.x == 0) des_v = atomicAdd(n_done, 1);
    des_v = __shfl_sync(0xffffffff, des_v, 0, blockDim.x);

    while (des_v < n_vertex) {
        scalar_t ret;

        int s_pos = col_starts[des_v];
        int e_pos = col_starts[des_v+1];

        scalar_t deg = 1.0 / (e_pos - s_pos);
        scalar_t * des_p = result + des_v * fsize;
        for (int k = threadIdx.x; k < fsize; k += blockDim.x) {
            ret = 0.0;
            for (int i = s_pos; i < e_pos; ++i) {
                ret += __ldg(features + rows[i] * fsize + k);
            }
            des_p[k] = ret * deg;
        }

        if (threadIdx.x == 0)
            des_v = atomicAdd(n_done, 1);
        des_v = __shfl_sync(0xffffffff, des_v, 0, blockDim.x);
    }
}

std::vector<torch::Tensor> gcn_conv_cuda_forward(
        torch::Tensor features,
        torch::Tensor col_starts,
        torch::Tensor rows) {
    auto result = torch::empty_like(features);

    const int n_vertex = features.size(0);
    const int feature_size = features.size(1);

    const int xaxis = (feature_size <= 16 ? 16 : 32);
    const int yaxis = 256 / xaxis;
    const dim3 threads(xaxis, yaxis);

    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "aggregate_forward_cuda", ([&] {
                int * n_done;
                cudaMalloc((void **)&n_done, sizeof(int));
                cudaMemset((void *)n_done, 0, sizeof(int));
                gcn_conv_cuda_forward_kernel<<<blocks, threads>>>(
                        n_done,
                        n_vertex,
                        feature_size,
                        features.data<scalar_t>(),
                        col_starts.data<int>(),
                        rows.data<int>(),
                        result.data<scalar_t>()
                        );
                cudaFree((void *)n_done);
                }));

    return {result};
}

template <typename scalar_t>
__global__ void gcn_conv_cuda_backward_kernel(
        int n_vertex,
        scalar_t * grad,
        scalar_t * indegs,
        int * row_starts,
        int * cols,
        scalar_t * result) {

    int src_v = blockIdx.x * blockDim.y + threadIdx.y;
    if (src_v < n_vertex)
    {
        scalar_t ret = 0.0;
        int s_pos = row_starts[src_v];
        int e_pos = row_starts[src_v+1];
        scalar_t * src_p = result + src_v * blockDim.x; 
        for (int i = s_pos; i < e_pos; ++i)
        {
            int des_v = cols[i];
            scalar_t * des_p = grad + des_v * blockDim.x;
            ret += des_p[threadIdx.x] / indegs[des_v];
        }
        src_p[threadIdx.x] = ret;
    }
}

std::vector<torch::Tensor> gcn_conv_cuda_backward(
        torch::Tensor features,
        torch::Tensor grad,
        torch::Tensor indegs,
        torch::Tensor row_starts,
        torch::Tensor cols) {
    auto result = torch::empty_like(features);

    const int w = 4;
    const int feature_size = features.size(1);

    const dim3 threads(features.size(1), w);
    const int blocks = (features.size(0) + w - 1) / w;

    // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "aggregate_forward_cuda", ([&] 
    {
        gcn_conv_cuda_backward_kernel<<<blocks, threads>>>(
            features.size(0),
            grad.data<scalar_t>(),
            indegs.data<scalar_t>(),
            row_starts.data<int>(),
            cols.data<int>(),
            result.data<scalar_t>()
        );
    }));

    return {result};
}
