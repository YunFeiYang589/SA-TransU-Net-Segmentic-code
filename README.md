# SA-TransU²Net: Rock Thin Section Grain Segmentation Network

**Official PyTorch Implementation**

> **SA-TransU²Net: A Rock Thin Section Grain Segmentation Network Based on Multi-scale RSU and Global Context Enhancement**  
> *Yaohua Gong, Di Shi, Ling Zhao, Yan Zhang, Lanyanlin Qu, Xiangrui Hou, Chengwu Xu, Juntao Gao, Yue Zhou, Zhiguo Wang*  
> School of Computer & Information Technology, Northeast Petroleum University, China  
> Corresponding Author: mirror_zl@163.com

---

## 📌 Overview

SA-TransU²Net is a hybrid deep learning model designed for high‑precision segmentation of sandstone thin‑section grains. It integrates a U²‑Net backbone with Swin‑Transformer self‑attention and a parameter‑free Spatial Attention (SA) mechanism to overcome challenges such as blurred grain boundaries, severe grain adhesion, and large scale variations.

The model achieves **87.68% mIoU**, **91.73% Precision**, and **92.66% Recall** on the Ordos Basin sandstone thin‑section dataset, significantly outperforming mainstream segmentation models.

---

## ✨ Key Features

- **Multi‑scale Residual U‑blocks (RSU)** – Nested U‑structures capture local context at multiple scales without exploding model depth.
- **Swin‑based Self‑Attention** – Introduced in deep encoder layers to model long‑range dependencies and separate adhered grains.
- **Parameter‑free Spatial Attention (SA)** – Recalibrates feature responses to boundaries and salient regions without extra learnable parameters.
- **Multi‑scale Deep Supervision** – Six side outputs plus a fused output for stable training and boundary preservation.
- **State‑of‑the‑art performance** – Outperforms SegFormer, SegNet, and U²‑Net on sandstone thin‑section images.

---

## 🧠 Architecture Overview

![SA-TransU²Net Architecture](docs/architecture.png)  
*(Refer to Fig. 1 in the paper for a detailed diagram.)*

- **Encoder**: RSU‑7 blocks in shallow/middle layers; Swin Transformer blocks (W‑MSA and SW‑MSA) in deep layers.
- **Decoder**: Progressive upsampling with skip connections and six side outputs.
- **Attention**: Parameter‑free SA modules embedded on the residual path of each RSU.
- **Fusion**: 1×1 convolution fuses multi‑scale side outputs for final prediction.

---

## 📁 Repository Structure

```
SA-TransU2Net/
├── models/
│   ├── sa_transunet.py          # Main network
│   ├── rsu.py                   # RSU and RSU4F modules
│   ├── swin_transformer.py      # Swin Transformer blocks & feature adaptation
│   └── spatial_attention.py     # Parameter‑free SA module
├── datasets/
│   ├── dataset.py               # Data loader with NLM preprocessing
│   └── transforms.py
├── train.py                     # Training script
├── test.py                      # Evaluation script
├── utils/
│   ├── metrics.py               # mIoU, Precision, Recall, Dice, Boundary IoU
│   └── losses.py                # BCE + Dice loss
├── configs/
│   └── default.yaml             # Hyperparameters
└── README.md
```

---

## 🚀 Getting Started

### Environment

- Python ≥ 3.9
- PyTorch ≥ 1.12
- CUDA 11.3+ (recommended)

Install dependencies:
```bash
pip install -r requirements.txt
```

Main packages: `torch`, `torchvision`, `numpy`, `opencv-python`, `pillow`, `scikit-image`, `tensorboard`, `pyyaml`.

---

### Dataset Preparation

Organise your data as:
```
data/
├── train/
│   ├── images/   (e.g., 001.png)
│   └── masks/    (binary PNG, same filename)
├── val/
└── test/
```

Our preprocessing pipeline applies Non‑Local Means denoising (search=21×21, patch=7×7, h=10) and crops 512×512 patches with stride 256 automatically. The dataset used in the paper (Ordos Basin sandstone) is available upon request from the corresponding author.

---

## 🏋️ Training

Start training with:
```bash
python train.py --config configs/default.yaml
```

Key hyperparameters (from the paper):
- Batch size: **16**
- Epochs: **400**
- Initial learning rate: **1e‑4** (with warmup + cosine annealing)
- Optimizer: **AdamW** (weight decay 1e‑4)
- Loss: **BCE + Dice** (equal weights)

The best model is saved based on validation mIoU.

---

## 📊 Evaluation

Run evaluation on the test set:
```bash
python test.py --weights path/to/checkpoint.pth --data_dir data/test
```

Reported metrics:
- **mIoU**, **Precision**, **Recall**, **Dice**
- **Boundary IoU** (5‑pixel band)
- Per‑size‑group Dice (small / medium / large grains)

---

## 📈 Results

### 1. Comparison with State‑of‑the‑Art Models (Table 3)

| Model          | mIoU (%)          | Precision (%)     | Recall (%)        | Dice (%)          |
|----------------|-------------------|-------------------|-------------------|-------------------|
| SegFormer      | 63.06 ± 0.42      | 79.17 ± 0.38      | 70.25 ± 0.38      | 88.26 ± 0.21      |
| SegNet         | 71.83 ± 0.38      | 85.45 ± 0.25      | 75.30 ± 0.22      | 83.54 ± 0.33      |
| U²‑Net         | 69.89 ± 0.25      | 81.90 ± 0.32      | 78.96 ± 0.25      | 81.23 ± 0.28      |
| **SA‑TransU²Net** | **87.68 ± 0.19** | **91.73 ± 0.20**  | **92.66 ± 0.22**  | **92.19 ± 0.17**  |

> Values are mean ± standard deviation over three independent runs (random seeds 42, 123, 999).

---

### 2. Ablation Study (Table 4)

| Configuration                 | Precision | mIoU   | Dice   | Recall |
|-------------------------------|-----------|--------|--------|--------|
| U²‑Net (baseline)             | 0.8190    | 0.6989 | 0.8041 | 0.7896 |
| U²‑Net + SA                   | 0.8575    | 0.8087 | 0.8623 | 0.8672 |
| U²‑Net + Swin‑Transformer     | 0.8627    | 0.7785 | 0.8476 | 0.8330 |
| **SA‑TransU²Net (full)**      | **0.9173** | **0.8768** | **0.9219** | **0.9266** |

---

### 3. Performance on Different Grain Sizes (Table 5 – Dice scores)

| Model          | Small grains (<5000 px²) | Medium grains (5000–20000 px²) | Large grains (>20000 px²) |
|----------------|--------------------------|--------------------------------|---------------------------|
| U²‑Net         | 0.712                    | 0.801                          | 0.825                     |
| **SA‑TransU²Net** | **0.864**                | **0.912**                      | **0.933**                 |

---

### 4. Boundary IoU

| Model          | Boundary IoU (%) |
|----------------|------------------|
| U²‑Net         | 68.5             |
| SegNet         | 71.0             |
| **SA‑TransU²Net** | **83.2**         |

---

### 5. Adhered Region Errors (20 challenging patches)

| Model          | Under‑segmentation errors | Over‑segmentation errors |
|----------------|---------------------------|--------------------------|
| U²‑Net         | 12                        | 8                        |
| **SA‑TransU²Net** | **3**                     | **2**                    |

---

## 🔧 Customisation

- **Input size**: Default 512×512; adjust in config (ensure compatibility with Swin patch embedding).
- **RSU depth**: RSU‑7 for all blocks, RSU4F for deep layers (modify `models/rsu.py`).
- **Swin parameters**: Window size = 7; attention heads = [2,4,8,16].
- **Loss weights**: Tune `alpha` and `beta` in `utils/losses.py`.

---

## 📝 Citation

If you use this code or find it helpful, please cite our paper:

```bibtex
@article{gong2025satransunet,
  title={SA-TransU²Net: A Rock Thin Section Grain Segmentation Network Based on Multi-scale RSU and Global Context Enhancement},
  author={Gong, Yaohua and Shi, Di and Zhao, Ling and Zhang, Yan and Qu, Lanyanlin and Hou, Xiangrui and Xu, Chengwu and Gao, Juntao and Zhou, Yue and Wang, Zhiguo},
  journal={IEEE Transactions on Geoscience and Remote Sensing},  % or the actual journal
  year={2025},
  note={In press / Accepted}
}
```

---

## 📄 License

This project is released under the MIT License. See `LICENSE` for details.

---

## 📧 Contact

For any questions regarding the code or the paper, please contact the corresponding author: **Ling Zhao** (mirror_zl@163.com).
