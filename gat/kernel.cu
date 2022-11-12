#include <vector>
#include <cstdio>
#include <cmath>

const unsigned short grain = 1;

template <typename scalar_t>
__global__ void gat_conv_cuda_forward_kernel(
        const int n_vertex,
        const int n_heads,
        const int fsize,
        const float relu_alpha,
        scalar_t * features,
        scalar_t * el,
        scalar_t * er,
        int * col_starts,
        int * rows,
        scalar_t * result) {

    int des_v;

    // if(threadIdx.x == 0) des_v = atomicAdd(n_done, grain);
    // des_v = __shfl_sync(0xffffffff, des_v, 0);
    // unsigned short step = 0;

    des_v = blockIdx.x * blockDim.y + threadIdx.y;

    // while (des_v < n_vertex)
    if (des_v < n_vertex)
    {
        int s_pos = col_starts[des_v];
        int e_pos = col_starts[des_v+1];
        scalar_t * des_p = result + des_v * fsize * n_heads;
        scalar_t * des_ep = er + des_v * n_heads;

        // head first loop
        for (int h = 0; h < n_heads; ++h) {
            scalar_t ret = 0.0;
            scalar_t det = 0.0;
            for (int k = threadIdx.x; k < fsize; k += blockDim.x) {
                for (int i = s_pos; i < e_pos; ++i)
                {
                    int src_v = rows[i];
                    scalar_t *src_p = features + (src_v * n_heads + h) * fsize;
                    scalar_t *src_ep = el + src_v * n_heads;

                    scalar_t cft = src_ep[h] + des_ep[h];
                    cft = (cft > 0.0 ? cft : relu_alpha * cft);
                    cft = expf(cft);
                    ret += cft * src_p[k];
                    det += cft;
                }
                des_p[k] = ret / det;
            }
            des_p += fsize;
        }

        // step++;
        // if(step == grain) 
        // {
        //     if(threadIdx.x == 0) des_v = atomicAdd(n_done, grain);
        //     des_v = __shfl_sync(0xffffffff, des_v, 0);
        //     step = 0;
        // }
        // else
        //     des_v++;

        // if(threadIdx.x == 0 && threadIdx.y == 0) 
        //     sharedm[0] = atomicAdd(n_done, blockDim.y);
        // __syncthreads();
        // des_v = sharedm[0] + threadIdx.y;

        // edge first loop
        // for (int i = s_pos, i < e_pos; ++i) {
        //     int src_v = rows[i];

        //     scalar_t * src_p = features + src_v * n_head * blockDim.x;
        //     scalar_t * src_ep = el + src_v * n_head;

        //     for (int h = 0; h < n_heads; ++h) {
        //         scalar cft = src_ep[h] + des_ep[h];
        //         cft = (cft > 0.0 ? cft : relu_alpha * cft);
        //         cft = expf(cft);
        //         des_p[h*n_heads+threadIdx.x] += src_p[threadIdx.x] * cft;
        //         sr[h] += cft;
        //         src_p += blockDim.x;
        //     }
        // }

        // for (int h = 0; i < n_heads; ++h) {
        //     des_p[h*n_heads+threadIdx.x] /= sr[h];
        // }
    }
}

std::vector<torch::Tensor> gat_conv_cuda_forward(
        torch::Tensor features,
        torch::Tensor el,
        torch::Tensor er,
        torch::Tensor col_starts,
        torch::Tensor rows) {
    auto result = torch::empty_like(features);

    const int w = 4;
    const int feature_size = features.size(2);
    const int n_heads = features.size(1);
    const int n_vertex = features.size(0);

    const int xaxis = (feature_size <= 16 ? 16 : 32);
    const dim3 threads(xaxis, w);
    // const int blocks = 640;
    const int blocks = (n_vertex + w - 1) / w;

    // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "aggregate_forward_cuda", ([&] 
    {
        // int * n_done;
        // cudaMalloc((void **)&n_done, sizeof(int));
        // cudaMemset((void *)n_done, 0, sizeof(int));
        gat_conv_cuda_forward_kernel<<<blocks, threads>>>(
            // n_done,
            n_vertex,
            n_heads,
            feature_size,
            0.2,
            features.data_ptr<scalar_t>(),
            el.data_ptr<scalar_t>(),
            er.data_ptr<scalar_t>(),
            col_starts.data_ptr<int>(),
            rows.data_ptr<int>(),
            result.data_ptr<scalar_t>()
        );
        // cudaFree((void *)n_done);
    }));
    return {result};
}

// template <typename scalar_t>
// __global__ void gat_conv_cuda_backward_kernel(
//         int n_vertex,
//         int n_heads,
//         float relu_alpha,
//         scalar_t * features,
//         scalar_t * el,
//         scalar_t * er,
//         scalar_t * grad,
//         int * col_starts,
//         int * rows,
//         scalar_t * result) {
//     int des_v = blockIdx.x * blockDim.y + threadIdx.y;
// 
//     if (des_v < n_vertex)
//     {
//         int s_pos = col_starts[des_v];
//         int e_pos = col_starts[des_v+1];
//         scalar_t * des_p = result + des_v * blockDim.x * n_heads;
//         scalar_t * des_ep = er + des_v * n_heads;
// 
//         // head first loop
//         for (int h = 0; h < n_heads; ++h) {
//             scalar_t ret = 0.0;
//             scalar_t det = 0.0;
// 
//             for (int i = s_pos; i < e_pos; ++i)
//             {
//                 int des_v = cols[i];
//                 scalar_t * des_p = grad + (des_v * n_heads + h) * blockDim.x;
//                 scalar_t * des_ep = er + des_v * n_heads;
// 
//                 scalar_t cft = src_ep[h] + des_ep[h];
//                 cft = (cft > 0.0 ? cft : relu_alpha * cft);
//                 cft = expf(cft);
//                 ret += cft * des_p[threadIdx.x];
//                 det += cft;
//             }
// 
//             src_p[threadIdx.x] = ret / det;
//             src_p += blockDim.x;
//         }
//     }
// 
// }
// 
// 
// std::vector<torch::Tensor> gat_conv_cuda_backward(
//         torch::Tensor features,
//         torch::Tensor el,
//         torch::Tensor er,
//         torch::Tensor grad,
//         torch::Tensor row_starts,
//         torch::Tensor cols) {
//     auto result = torch::empty_like(features);
// 
//     const int w = 8;
//     const int feature_size = features.size(2);
//     const int n_heads = features.size(1);
//     const int n_vertex = features.size(0);
// 
//     const dim3 threads(feature_size, w);
//     const int blocks = (n_vertex + w - 1) / w;
// 
//     AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "aggregate_backward_cuda", ([&] 
//     {
//         gat_conv_cuda_backward_kernel<<<blocks, threads>>>(
//             n_vertex,
//             n_heads,
//             0.2,
//             features.data_ptr<scalar_t>(),
//             el.data_ptr<scalar_t>(),
//             er.data_ptr<scalar_t>(),
//             col_starts.data_ptr<int>(),
//             rows.data_ptr<int>(),
//             result.data_ptr<scalar_t>()
//         );
//     }));
//     return {result};
// }
