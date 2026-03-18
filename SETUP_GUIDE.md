# DINOv2 Setup Guide — Step-by-Step

This guide walks you through setting up DINOv2 manually for spot-guided reidentification, from environment creation to running your first inference.

---

## Prerequisites (Before You Start)

| Requirement | What You Need |
|-------------|---------------|
| **Python** | 3.9 or 3.10 (recommended for compatibility) |
| **CUDA** | 11.7+ if using GPU (check with `nvidia-smi`) |
| **Disk space** | ~2–4 GB for model weights (first load downloads automatically) |
| **RAM** | 8 GB minimum; 16 GB recommended for ViT-L/g |

---

## Step 1: Create a Virtual Environment

Choose **one** of these methods:

### Option A: venv (Python built-in)

```bash
cd /home/fog/astr_research
python3 -m venv .venv
source .venv/bin/activate   # On Linux/Mac
```

### Option B: conda (recommended if you have it)

```bash
conda create -n dinov2 python=3.9
conda activate dinov2
```

---

## Step 2: Install PyTorch with CUDA (or CPU)

DINOv2 loads via PyTorch Hub. PyTorch is the only hard dependency for inference.

### With CUDA 11.8 (recommended for GPU)

```bash
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
```

### With CUDA 12.1

```bash
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu121
```

### CPU only

```bash
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu
```

Verify installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Step 3: Install Optional Dependencies

For the OOP wrapper and demo scripts (image loading, NumPy, etc.):

```bash
pip install -r requirements.txt
```

This installs: ` Pillow `, ` numpy `, ` tqdm `. These are for the demo; DINOv2 itself only needs PyTorch.

---

## Step 4: Verify DINOv2 Loads via PyTorch Hub

```bash
python -c "
import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
print('DINOv2 loaded successfully!')
print(f'Model: {type(model).__name__}')
"
```

First run will download model weights (~90 MB for ViT-S). Subsequent runs use cache.

---

## Step 5: Project Layout

After setup, your layout looks like:

```
astr_research/
├── .venv/                    # or conda env
├── astr_vs_loftr.pdf
├── SETUP_GUIDE.md            # this file
├── requirements.txt
├── dinov2_extractor.py       # OOP feature extractor
├── demo_spot_reid.py         # demo script
└── dinov2/                   # (optional) cloned repo for training/notebooks
```

---

## Step 6: Clone DINOv2 Repo (Optional)

Only needed if you want notebooks, training, or eval scripts. **Inference works without this.**

```bash
git clone https://github.com/facebookresearch/dinov2.git
cd dinov2
```

For **training/evaluation**, install full deps:

```bash
pip install -r requirements.txt
# Or with conda:
# conda env create -f conda.yaml
# conda activate dinov2
```

---

## Step 7: Run the Demo

```bash
cd /home/fog/astr_research
source .venv/bin/activate   # or: conda activate dinov2
python demo_spot_reid.py
```

The demo will:

1. Load DINOv2 (ViT-S by default)
2. Create dummy images or use paths you provide
3. Extract patch features from image A and image B
4. Compute cosine similarity between a chosen spot in A and all patches in B
5. Report the best-matching patch location

---

## Model Variants

| Model | Params | Speed | Accuracy | Use Case |
|-------|--------|-------|----------|----------|
| `dinov2_vits14` | 21M | Fast | Good | Quick prototyping |
| `dinov2_vitb14` | 86M | Medium | Better | Balanced |
| `dinov2_vitl14` | 300M | Slow | Best | Max accuracy |
| `dinov2_vitg14` | 1.1B | Slowest | Highest | Research only |

Load via: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Use `dinov2_vits14` or reduce batch size |
| `xformers` / `cuml` errors | You don't need these for inference; use minimal `requirements.txt` only |
| `torch.hub` download fails | Set `HF_HOME` or `TORCH_HOME` if behind a proxy |
| Slow first load | Weights download once; subsequent loads are fast |

---

## Quick Reference: End-to-End Commands

```bash
cd /home/fog/astr_research
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -c "import torch; m=torch.hub.load('facebookresearch/dinov2','dinov2_vits14'); print('OK')"
python demo_spot_reid.py
```
