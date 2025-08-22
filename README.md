---
# MAGNeT: Multimodal Adaptive Gaussian Networks for Intent Inference in Moving Target Selection across Complex Scenarios

This repository contains the PyTorch implementation of the **Multimodal Adaptive Gaussian Network (MAGNeT)**, designed for moving target selection.

---
## Environment Setup

To get started, ensure you have the following installed:

* **Python**: Version 3.10 or newer
* **PyTorch**: Version 2.4.0 or newer
* **Scikit-learn**: Version 1.5.2 or newer
* **Pandas**: Version 2.2.2 or newer

MAGNeT is computationally efficient and can run on CPU/GPU with at least 2GB of memory. Our experiments were conducted on an **RTX 3090 (24GB GPU)**, utilizing approximately **1.8GB of memory** for a batch size of 32. This setup achieved an impressive processing speed of $2.84 \times 10^{-4}$ seconds per sample during training and $1.43 \times 10^{-4}$ seconds during testing.

---
## Dataset Preparation

To use the provided datasets, follow these steps:

1. **Download the dataset package**:
   - You can download the compressed dataset package [`
MTS_dataset.zip`](https://drive.google.com/file/d/1cpu6RFmb4LDOOBzzF9SjO111VdlD6Mqu/view?usp=sharing).
   
2. **Extract and place the folders**:
   - After downloading, extract the `MTS_dataset.zip` file.
   - Move the extracted `dataset_2d` and `dataset_3d` folders to the same directory as your `fusion_file_2d.py` or `fusion_file_3d` file.

3. **Using your own datasets** (optional):
   - If you prefer to prepare your own datasets, you can refer to the structure and format of the provided `dataset_2d` and `dataset_3d` folders as a reference.

Once the datasets are properly placed, you should be able to run the code without additional configuration.

---
## Training and Testing

We use 5 randomly generated seeds for dataset splitting and weight initialization. The random numbers will be generated in `random_number.npy` file automatically.

To train and test MAGNeT, execute the following command:

For 2D moving target selection:

```bash
python fusion_file_2d.py --use_env --epochs 50 --num_shot [NUM_FEW_SHOT] --result_path [PATH_TO_SAVE]
```

For 3D moving target selection:

```bash
python fusion_file_3d.py --use_env --epochs 50 --num_shot [NUM_FEW_SHOT] --result_path [PATH_TO_SAVE]
```

Detailed parameter settings can be found in parse.py.

# Reference

```latex
@article{li2025magnet,
  title={MAGNeT: Multimodal Adaptive Gaussian Networks for Intent Inference in Moving Target Selection across Complex Scenarios},
  author={Li, Xiangxian and Zheng, Yawen and Zhang, Baiqiao and Ma, Yijia and XianhuiCao, XianhuiCao and Liu, Juan and Bian, Yulong and Huang, Jin and Yang, Chenglei},
  journal={arXiv preprint arXiv:2508.12992},
  year={2025}
}
```
