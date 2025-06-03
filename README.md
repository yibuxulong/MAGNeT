---
# MAGNeT: Multimodal Adaptive Gaussian Network for Moving Target Selection

This repository contains the PyTorch implementation of the **Multimodal Adaptive Gaussian Network (MAGNeT)**, designed for moving target selection.

---
## Environment Setup

MAGNeT is computationally efficient and can run on GPUs with at least 2GB of memory. Our experiments were conducted on an **RTX 3090 (24GB GPU)**, utilizing approximately **1.8GB of memory** for a batch size of 32. This setup achieved an impressive processing speed of $2.84 \times 10^{-4}$ seconds per sample during training and $1.43 \times 10^{-4}$ seconds during testing.

To get started, ensure you have the following installed:

* **Python**: Version 3.10 or newer
* **PyTorch**: Version 2.4.0 or newer
* **Scikit-learn**: Version 1.5.2 or newer
* **Pandas**: Version 2.2.2 or newer

---
## Training and Testing

We use 5 randomly generated seeds for dataset splitting and weight initialization. If you wish to use your own seeds, simply replace the `random_number.npy` file.

To train and test MAGNeT, execute the following command:

```bash
python fusion_file.py --use_env --epochs 50 --num_shot [NUM_FEW_SHOT] --result_path [PATH_TO_SAVE]
```

Detailed parameter settings can be found in parse.py.

Note: We will release the MTS-2D and MTS-3D datasets, along with the code for 3D training and testing, once our paper has been accepted.
