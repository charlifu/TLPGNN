import argparse, time
import numpy as np
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.profiler as profiler
import scipy.sparse as sp
from torch.utils.cpp_extension import load_inline

def read_data(dataset):
    data_path = "../data/" + dataset + "/"
    ret = {}
    ret['features'] = np.load(data_path+'features.npy')
    ret['graph'] = sp.load_npz(data_path+'csr.npz').tocsc()
    ret['graph'].sort_indices()
    return ret

cpp_source = '''
#include <vector>

std::vector<torch::Tensor> gat_conv_cuda_forward(
        torch::Tensor features,
        torch::Tensor el,
        torch::Tensor er,
        torch::Tensor col_starts,
        torch::Tensor rows);

std::vector<torch::Tensor> gat_conv_cuda_backward(
        torch::Tensor features,
        torch::Tensor el,
        torch::Tensor er,
        torch::Tensor grad,
        torch::Tensor row_starts,
        torch::Tensor cols);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> gat_conv_forward(
        torch::Tensor features,
        torch::Tensor el,
        torch::Tensor er,
        torch::Tensor col_starts,
        torch::Tensor rows)
{
    CHECK_INPUT(features);
    CHECK_INPUT(col_starts);
    CHECK_INPUT(rows);
    CHECK_INPUT(el);
    CHECK_INPUT(er);

    return gat_conv_cuda_forward(features, el, er, col_starts, rows);
}

// std::vector<torch::Tensor> gat_conv_backward(
//         torch::Tensor features,
//         torch::Tensor el,
//         torch::Tensor er,
//         torch::Tensor grad,
//         torch::Tensor row_starts,
//         torch::Tensor cols)
// {
//     CHECK_INPUT(features);
//     CHECK_INPUT(row_starts);
//     CHECK_INPUT(cols);
//     CHECK_INPUT(el);
//     CHECK_INPUT(er);
//     CHECK_INPUT(grad);
//
//     return gat_conv_cuda_backward(features, el, er, grad, row_starts, cols);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gat_conv_forward, "GAT aggregate forward (CUDA)");
    // m.def("backward", &gat_conv_backward, "GAT aggregate backward (CUDA)");
}
'''

cuda_source = open("kernel.cu").read()

gat_module = load_inline(name="gat",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        extra_cuda_cflags=['-m 64'],
        verbose=False)


def main(args):
    th.cuda.set_device(args.gpu)

    # load and preprocess dataset
    data = read_data(args.dataset)
    indptr = data['graph'].indptr
    col_starts = th.cuda.IntTensor(indptr)
    rows = th.cuda.IntTensor(data['graph'].indices)

    features = th.cuda.FloatTensor(data['features'][:,0:args.size*args.heads]).view(-1, args.heads, args.size)
    el = th.cuda.FloatTensor(data['features'][:,10:10+args.heads])#.unsqueeze(-1)
    er = th.cuda.FloatTensor(data['features'][:,20:20+args.heads])#.unsqueeze(-1)

    rst = gat_module.forward(features, el, er, col_starts, rows)
    th.cuda.synchronize()

    run_time = 0.0
    for _ in range(10):
        start_run = time.perf_counter()
        gat_module.forward(features, el, er, col_starts, rows)
        th.cuda.synchronize()
        run_time += (time.perf_counter() - start_run)

    print('Time (ms): {:.3f}'.format(run_time*1e3/10))
    return '{:.3f}'.format(run_time*1e3/10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
            default="citeseer",
            help="which graph data set to process")
    parser.add_argument("-s", "--size",
            help="feature size", type=int, default=32)
    parser.add_argument("-g", "--gpu",
            type=int, default=0, help="gpu")
    parser.add_argument("-H", "--heads",
            type=int, default=1, help="number of heads")
    args = parser.parse_args()
    print(args)

    main(args)
