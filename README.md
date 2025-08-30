# Deepfake-Detector

Two-branch image classifier for deepfake detection on face crops.

- **RGB branch** — Swin-Tiny (3-channel images)
- **Frequency branch** — EfficientNet-B0 (single-channel FFT amplitude map with low-frequency suppression)

Branches are concatenated and logits are obtained using two heads:
- **Binary head** — real vs fake  
- **Type head** — deepfake-generator class (multiclass, in the given dataset there were 7 classes - 6 for various generator types and 1 for real image)

---

## Repository Structure

- `data.py` — dataset, transforms, FFT pipeline, loaders, sampler & split  
- `model.py` — backbones and Lightning module  
- `train.py` — the main script containing training/validation/prediction + auto data download  
- `fft_imgs_example.ipynb` — example of FFT channel creation  
- `.gitignore`, `README.md`

---

## Data Format
In case if you would like to use your own dataset:

**Training CSV** (expected at `./train/train.csv`) must include:
- `crop_path` — relative image path inside `train/`
- `label` — `0` for fake, `1` for real
- `fake_type` — string category of the deepfake-generator or just string 'real' if label is 1

**Test CSV** (expected at `./test/test.csv`) must include:
- `crop_path` — relative image path inside `test/`

---

## Installation

Use Python **3.9–3.11**.

```bash
# Install PyTorch/torchvision suitable for your CUDA/CPU:
pip install torch torchvision

# Core dependencies:
pip install pytorch-lightning torchmetrics timm pandas numpy requests tensorboard thop
```

---

## Quickstart

```bash
# 1) Clone
git clone https://github.com/quantumoon/deepfake-detector.git
cd deepfake-detector

# 2) Train (script will download data if missing)
python train.py
```

Outputs:
- TensorBoard logs: `tb_logs/deepfake_exp/`
- Best checkpoint by `val/roc_auc`
- `submission.csv` with probabilities for the test split
