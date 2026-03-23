# PreBackRazor (Sheared Backpropagation for Fine-tuning Foundation Models, Core Algorithm Release)

This repository was automatically organized using an AI model.

This repository is a cleaned, core-algorithm-focused release for static code reading.
It keeps the essential implementation of sparse backpropagation and the selective backpropagation
policy used by PreBackRazor, while removing training/data pipelines and duplicated backup/copy files.

If you are here to understand the algorithm (not to run full experiments), start with:

- `custom_functions/custom_fc.py`: where the selective `prebackrazor` decision happens
- `custom_functions/sparse_matrix.py`: `sparsify()` / `unsparsify()` for sparse backward storage
- `ViT/models/modeling.py` + `ViT/models/modeling_new_prune.py`: how the sparse operators are wired into ViT

We intentionally provide a “core snapshot” so the important ideas are easy to read without forcing readers
to make the full training pipeline run end-to-end.

## Acknowledgements to BackRazor

This work is built on the BackRazor codebase and its core idea of self-sparsified backpropagation.
We would like to thank the BackRazor authors and the VITA-Group team for open-sourcing BackRazor:
[`VITA-Group/BackRazor_Neurips22`](https://github.com/VITA-Group/BackRazor_Neurips22).

If it were not for this excellent work, our project would not have been able to start.

Many design elements in this repository (including the sparse-backward storage concept and module structure)
follow BackRazor closely. Where we changed details to support PreBackRazor-style selective behavior,
those changes are described in `THIRD_PARTY_NOTICES.md`.

We are grateful for the public resources released by the community, and we hope this core release makes it
easy for readers to focus on the algorithmic contribution rather than debugging the full training setup.

This started as a student project (years ago) by the author, and many configurations in the original training
pipeline are now outdated. However, the activation sparsity direction—especially column-wise and
norm-induced sparsity—still looks promising.

## Scope of This Release

- Keep only core algorithm code for static inspection.
- Keep proper citation and credits to prior work, especially BackRazor.
- Describe key modifications introduced in this work.
- This snapshot does not bundle third-party utility library source code such as `timm`.

This repository is not intended to be a full reproduction pipeline.

## Core Algorithm Map

### 1) Selective backpropagation (PreBackRazor policy)

Main file: `custom_functions/custom_fc.py`

- `LinearSparse.forward()` routes between two backward strategies:
  - `linear_prebackrazor` when `prebackrazor == True` and current weight does not require grad.
  - `linear_postbackrazor` otherwise.
- This is the central switch for reducing backward overhead on selected layers.

### 2) Activation sparsification for backward storage

Main file: `custom_functions/sparse_matrix.py`

- `sparsify()` compresses masked activations into sparse values + packed bit mask.
- `unsparsify()` reconstructs dense tensors in backward pass from saved sparse states.
- This corresponds to the memory-saving mechanism inherited/adapted from BackRazor-style sparse backward storage.

### 3) ViT module integration

Main files: `ViT/models/modeling.py`, `ViT/models/modeling_new_prune.py`

- `modeling.py` defines the ViT structure and injects sparse operators.
- `modeling_new_prune.py` integrates sparse linear/matmul/softmax pathways and related attention-path logic.

## Third-Party Credits

This project builds on and adapts ideas/code from:

- BackRazor: [VITA-Group/BackRazor_Neurips22](https://github.com/VITA-Group/BackRazor_Neurips22)
- Mesa: [zhuang-group/Mesa](https://github.com/zhuang-group/Mesa)
- TinyTL: [mit-han-lab/tinyml](https://github.com/mit-han-lab/tinyml/tree/master/tinytl)
- ViT-pytorch: [jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)

Detailed module-level attribution and modification notes are provided in `THIRD_PARTY_NOTICES.md`.

Note: this core release is organized for algorithm readability. File-level attribution could be incomplete for some
source files derived from the upstream projects listed above; please review upstream repositories and licenses if you redistribute.

If you find missing pieces (for example, configuration details) when adapting this core release,
please refer to the original BackRazor codebase to fill in the required setup.

## License

This repository now uses the original license.
See `LICENSE` for details.

## Cite

### PreBackRazor (this work)

```bibtex
@InProceedings{Yu_2024_CVPR,
  author = {Yu, Zhiyuan and Shen, Li and Ding, Liang and Tian, Xinmei and Chen, Yixin and Tao, Dacheng},
  title = {Sheared Backpropagation for Fine-tuning Foundation Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2024},
  pages = {5883-5892}
}
```

Paper page: <https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Sheared_Backpropagation_for_Fine-tuning_Foundation_Models_CVPR_2024_paper.html>  
PDF: <https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Sheared_Backpropagation_for_Fine-tuning_Foundation_Models_CVPR_2024_paper.pdf>

### BackRazor (upstream inspiration/baseline)

```bibtex
@inproceedings{
jiang2022back,
title={Back Razor: Memory-Efficient Transfer Learning by Self-Sparsified Backpropogation},
author={Jiang, Ziyu and Chen, Xuxi and Huang, Xueqin and Du, Xianzhi and Zhou, Denny and Wang, Zhangyang},
booktitle={Advances in Neural Information Processing Systems 36},
year={2022}
}
```

BackRazor repository: <https://github.com/VITA-Group/BackRazor_Neurips22>
