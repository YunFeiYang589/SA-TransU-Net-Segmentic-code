# SA-TransU²Net: Rock Thin Section Grain Segmentation Network

This repository contains the official PyTorch implementation of **SA-TransU²Net**, a hybrid deep learning model for high-precision grain segmentation in sandstone thin-section images. The model integrates a U²-Net backbone with Swin‑Transformer self‑attention and a parameter‑free Spatial Attention (SA) mechanism to address challenges such as blurred grain boundaries, severe grain adhesion, and large scale variations.

> **Paper:** *SA-TransU²Net: A Rock Thin Section Grain Segmentation Network Based on Multi-scale RSU and Global Context Enhancement*  
> **Authors:** Yaohua Gong, Di Shi, Ling Zhao, Yan Zhang, Lanyanlin Qu, Xiangrui Hou, Chengwu Xu, Juntao Gao, Yue Zhou, Zhiguo Wang

---

## 📌 Key Features

- **Multi‑scale Residual U‑blocks (RSU)** – capture local context at multiple scales using nested U‑structures.
- **Swin‑based Self‑Attention** – introduced in the deep encoder layers to model long‑range dependencies and separate adhered grains.
- **Parameter‑free Spatial Attention (SA)** – enhances feature responses to grain boundaries and salient structures without extra learnable parameters.
- **Multi‑scale Deep Supervision** – six side outputs plus a fused output for stable training and boundary‑detail preservation.
- **State‑of‑the‑art performance** – achieves **87.68% mIoU**, **91.73% Precision**, and **92.66% Recall** on the Ordos Basin sandstone dataset.

---

## 📁 Repository Structure

```
SA-TransU2Net/
├── models/
│   ├── sa_transunet.py          # Main network definition
│   ├── rsu.py                   # RSU and RSU4F modules
│   ├── swin_transformer.py      # Swin Transformer blocks & feature adaptation
│   └── spatial_attention.py     # Parameter‑free SA module
├── datasets/
│   ├── dataset.py               # Dataset loader with NLM preprocessing
│   └── transforms.py            # Data augmentation utilities
├── train.py                     # Training script with deep supervision
├── test.py                      # Evaluation script (metrics & visualization)
├── utils/
│   ├── metrics.py               # mIoU, Precision, Recall, Dice, Boundary IoU
│   └── losses.py                # Combined BCE + Dice loss
├── configs/
│   └── default.yaml             # Hyperparameters (batch size, LR, epochs, etc.)
├── weights/                     # Pretrained model checkpoints
├── logs/                        # TensorBoard logs
└── README.md
```

---

## 🚀 Getting Started

### Environment Setup

- Python ≥ 3.9
- PyTorch ≥ 1.12
- CUDA 11.3+ (recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `torch`, `torchvision`, `numpy`, `opencv-python`, `pillow`, `scikit-image`, `tensorboard`, `pyyaml`.

---

### Dataset Preparation

1. **Dataset format** – Organize your images and masks as:
   ```
   data/
   ├── train/
   │   ├── images/   (e.g., 001.png)
   │   └── masks/    (binary PNG, same filename)
   ├── val/
   └── test/
   ```
2. **Preprocessing** – Our pipeline applies Non‑Local Means denoising (search=21×21, patch=7×7, h=10) and crops 512×512 patches with a stride of 256. The code handles this automatically if you place full‑resolution images in the respective folders.
3. For the **Ordos Basin sandstone dataset** used in the paper, please contact the corresponding author (mirror_zl@163.com) for access.

---

## 🧠 Model Architecture Overview

![SA-TransU²Net Architecture](docs/architecture.png)  
*(Refer to Fig. 1 in the paper for a detailed diagram.)*

- **Encoder**: RSU blocks (RSU‑7) in shallow/middle layers; Swin Transformer blocks (with W‑MSA and SW‑MSA) in deep layers.
- **Decoder**: Progressive upsampling with skip connections and deep side outputs (6 scales).
- **Attention**: Parameter‑free SA modules are embedded on the residual path of each RSU.
- **Fusion**: 1×1 convolution fuses multi‑scale side outputs for final prediction.

---

## 🏋️ Training

To train from scratch:

```bash
python train.py --config configs/default.yaml
```

Key hyperparameters (can be modified in the YAML file or passed as CLI args):
- Batch size: 16 (adjust based on GPU memory)
- Epochs: 400
- Learning rate: 1e‑4 with warmup (2 epochs) and cosine annealing
- Optimizer: AdamW (weight decay 1e‑4)
- Loss: BCE + Dice (equal weights)

The script logs training/validation losses, and saves the best model based on validation mIoU.

---

## 📊 Evaluation

To evaluate on the test set:

```bash
python test.py --weights path/to/checkpoint.pth --data_dir data/test
```

Metrics reported:
- **mIoU** (Mean Intersection over Union)
- **Precision**
- **Recall**
- **Dice Coefficient**
- **Boundary IoU** (within 5‑pixel band around ground truth boundaries)
- Per‑size‑group Dice (small / medium / large grains)

Visualization of segmentation masks will be saved in `results/` for qualitative inspection.

---

## 📈 Results

### Performance on Ordos Basin Sandstone Dataset

| Model          | mIoU (%) | Precision (%) | Recall (%) | Dice (%) |
|----------------|----------|---------------|------------|----------|
| SegFormer      | 63.06    | 79.17         | 70.25      | 88.26    |
| SegNet         | 71.83    | 85.45         | 75.30      | 83.54    |
| U²‑Net         | 69.89    | 81.90         | 78.96      | 81.23    |
| **SA‑TransU²Net** | **87.68** | **91.73**     | **92.66**  | **92.19** |

### Ablation Study

| Configuration                 | mIoU (%) | Precision (%) | Recall (%) | Dice (%) |
|-------------------------------|----------|---------------|------------|----------|
| U²‑Net (baseline)             | 69.89    | 81.90         | 78.96      | 81.23    |
| + SA module                   | 80.87    | 85.75         | 86.72      | 86.23    |
| + Swin‑Transformer            | 77.85    | 86.27         | 83.30      | 84.76    |
| **Full SA‑TransU²Net**        | **87.68** | **91.73**     | **92.66**  | **92.19** |

### Cross‑Dataset Generalization

The model was tested on unseen public datasets (volcanic rock, limestone) and achieved consistent boundary‑accurate segmentation, demonstrating strong robustness and the ability to detect grains missed by manual annotations.

---

## 🧪 Error Analysis (from paper)

- **Boundary IoU**: 83.2% vs. U²‑Net 68.5% and SegNet 71.0%.
- **Small‑grain Dice**: 86.4% vs. U²‑Net 71.2%.
- **Adhered‑region errors**: Only 3 under‑segmentation and 2 over‑segmentation errors across 20 challenging patches, compared to 12 and 8 for U²‑Net.

---

## 🔧 Customization

- **Input size**: The default patch size is 512×512; you can change it in the config (ensure it's compatible with Swin patch embedding).
- **RSU depth**: The paper uses RSU‑7 for all blocks; RSU4F for deep layers. Modify `models/rsu.py` if needed.
- **Swin parameters**: Window size = 7, attention heads per stage = [2,4,8,16].
- **Loss weights**: Adjust `alpha` and `beta` in `utils/losses.py` to prioritize BCE or Dice.

---

## 📝 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{gong2025satransunet,
  title={SA-TransU²Net: A Rock Thin Section Grain Segmentation Network Based on Multi-scale RSU and Global Context Enhancement},
  author={Gong, Yaohua and Shi, Di and Zhao, Ling and Zhang, Yan and Qu, Lanyanlin and Hou, Xiangrui and Xu, Chengwu and Gao, Juntao and Zhou, Yue and Wang, Zhiguo},
  journal={IEEE Transactions on Geoscience and Remote Sensing},  % or the actual journal
  year={2025},
  note={In preparation / Accepted}
}
```

---

## 📄 License

This project is released under the MIT License. See `LICENSE` for details.

---

## 🤝 Contributing

We welcome contributions! Please open an issue or submit a pull request for any improvements, bug fixes, or extensions.

---

## 📧 Contact

For questions regarding the code or the paper, please contact the corresponding author: **Ling Zhao** (mirror_zl@163.com).
