# Third-Party Notices and Credits

This repository includes ideas and partial implementations adapted from prior projects.
We thank the original authors.

## Upstream Projects

1. BackRazor
   - Repository: [https://github.com/VITA-Group/BackRazor_Neurips22](https://github.com/VITA-Group/BackRazor_Neurips22)
   - Paper: "Back Razor: Memory-Efficient Transfer Learning by Self-Sparsified Backpropogation" (NeurIPS 2022)
   - Role here: baseline design for sparse backward activation storage and related module structure.

2. Mesa
   - Repository: [https://github.com/zhuang-group/Mesa](https://github.com/zhuang-group/Mesa)
   - Role here: quantization/native kernel interfaces used by sparse custom operators.

3. TinyTL
   - Repository: [https://github.com/mit-han-lab/tinyml/tree/master/tinytl](https://github.com/mit-han-lab/tinyml/tree/master/tinytl)
   - Role here: selected transfer-learning/training design references in earlier code iterations.

4. ViT-pytorch
   - Repository: [https://github.com/jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
   - Role here: Vision Transformer code structure and loading utilities referenced/adapted.

## Main Modifications in This Repository

Compared to upstream baselines, this repository focuses on PreBackRazor-style selective backward behavior and cleaned core code:

- `custom_functions/custom_fc.py`
  - Added and used `linear_prebackrazor` path for cases where selected layer weights do not require gradient.
  - Kept `linear_postbackrazor` path for normal sparse-backward behavior.
  - `LinearSparse.forward()` contains the selection logic between the two paths.

- `custom_functions/sparse_matrix.py`
  - Uses packed mask + sparse values serialization (`sparsify`) and restoration (`unsparsify`) for backward memory saving.

- `custom_functions/custom_matmul.py`, `custom_functions/custom_softmax_matmul.py`
  - Sparse/quantized custom autograd operators are integrated for activation-saving backward flow.

- `ViT/models/modeling_new_prune.py` and `ViT/models/modeling.py`
  - Integrate sparse custom operators into ViT attention/MLP path.
  - Expose and pass `prebackrazor` behavior into sparse modules.

## Code Cleanup Policy in This Release

- Removed duplicate backup/copy files and non-core experiment artifacts.
- Removed training/data orchestration scripts from this release to keep a core algorithm snapshot.
- No `timm` source files are included in this repository snapshot.

## Note on `mesa` Imports

Several files still import `mesa` modules (for example `custom_quant`, `native`, `packbit`).
In many places this is primarily to keep interface compatibility with upstream implementations
and avoid introducing additional code changes while preparing this minimal core release.

## Important Licensing Note

This repository uses the original project license in `LICENSE`.
Third-party components may be subject to their own original licenses.
Users are responsible for checking and complying with upstream license terms when reusing or redistributing derived code.

## Attribution Caveat (File-Level Citations May Be Incomplete)

Because this release is a “core algorithm snapshot”, file-level provenance information may not be perfectly complete.
Some source files in this repository could be derived from (or adapted from) upstream code listed above,
even if we did not explicitly mention every such file individually in this document.

If you plan to reuse, redistribute, or publish derived works, please review the original upstream repositories
and their licenses, and (if needed) verify provenance for specific files or commit histories.
