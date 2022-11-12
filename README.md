
---
Software requirement
-----
gcc 8.5

CUDA 11.3

Pytorch 1.10

---
Hardware
------
GPU: V100 (tested)

----
Execute
------
```bash
cd gcn/
python test_kernel --dataset citeseer --size 32
```
To run TLPGNN on other dataset, please add the data folder under `data/` and provide the same data files with our citeseer example.

----
Reference
-------

If you use iSpan in your project, please cite the following paper.

```python
@inproceedings{fu2022tlpgnn,
  title={TLPGNN: A Lightweight Two-Level Parallelism Paradigm for Graph Neural Network Computation on GPU},
  author={Fu, Qiang and Ji, Yuede and Huang, H Howie},
  booktitle={Proceedings of the 31st International Symposium on High-Performance Parallel and Distributed Computing},
  pages={122--134},
  year={2022}
}
```