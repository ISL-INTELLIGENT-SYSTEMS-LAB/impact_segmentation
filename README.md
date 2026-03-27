# astr_research — DINOv2 setup & re-identification tooling

This README includes **environment setup** for DINOv2 and, under [Recent updates](#recent-updates-march-2026), the **SAM3 + DINOv2 spot re-ID pipeline** with **TorchMetrics** retrieval mAP, ground-truth labeling, and saved metrics.

---

## Recent updates (March 2026)

The following was added to the codebase:

- **`reidentification_pipeline/demo_spot_reid_v2_Copy.py`** — SAM3 (text prompt → boxes) + DINOv2 (cosine similarity per box pair) + optional **TorchMetrics `RetrievalMAP`** when you supply a ground-truth file.
- **`torchmetrics`** — listed in `requirements.txt`; used only when you pass `--gt_json` (see below).
- **Ground truth JSON (input, you author it)** — describes which **query detection** should match which **reference box index(es)**. The pipeline does **not** infer this; you provide the “answer key” for evaluation.
- **Metrics JSON (output, auto-generated)** — each run with `--gt_json` writes `copy_v2_results/metrics/metrics_YYYYMMDD_HHMMSS.json` (or under your `--output` directory) so runs do not overwrite each other.
- **Diagnostics** — if a ground-truth row cannot be evaluated (e.g. SAM found no object on a query image), the script prints the reason and may record **`skipped_ground_truth`** in the metrics JSON.

Example files live under `copy_v2_results/json_results/` (e.g. `ground_truth_mock.json` and `HOW_TO_USE_GROUND_TRUTH.txt`).

---

## SAM3 + DINOv2 spot re-ID and TorchMetrics

### What the pipeline does (no ground truth)

1. **`--ref_image_path`** — one reference image. SAM3 segments objects matching your text prompt (e.g. `"yellow ball"`). Each box gets index `0, 1, 2, …` in SAM’s order.
2. **`--image_directory`** — query images. SAM3 runs on each; DINOv2 builds a feature vector per box.
3. **Cosine similarity** — a matrix of shape `(num_reference_boxes × num_query_boxes)` is printed; segmented PNGs are saved under **`--output`**.

Without ground truth, there is **no** mAP—only similarities and visualizations.

### Why TorchMetrics / mAP?

**Mean Average Precision (RetrievalMAP)** scores **ranking quality**: for each **labeled query detection**, all reference boxes are ranked by DINO similarity. If the **true** reference index (from your labels) ranks high, AP is high; **mAP** averages that over all labeled queries.

This does **not** prove “same physical object” in the wild; it measures how well **your** DINO-based scores align with **your** labels.

### Ground truth JSON (input — you create this)

Pass **a path to a single `.json` file** (not a directory), e.g. `--gt_json /path/to/ground_truth.json`.

Format: a **JSON array** of rules. Each object describes **one query-side detection** you want to score:

| Field | Meaning |
|--------|--------|
| `query_image` | **Basename** of a file inside `--image_directory` (must match the filename exactly). |
| `query_box_idx` | Which SAM box on **that query image**: `0` = first box, `1` = second, … |
| `relevant_ref_indices` | Which box(es) on the **reference image** (`--ref_image_path`) are **correct** matches: `0` = first reference box, etc. Often `[0]` for a single ball. |

The reference image path is **not** repeated in this file; it comes from `--ref_image_path`. If you change query filenames in `temp_folder`, update `query_image` entries. If you change the reference image, you may need to change **`relevant_ref_indices`** if SAM’s box order or count changes.

**Not every query file must appear** in the JSON—only list rows for detections you want included in mAP. If SAM returns **zero** boxes on a query image, that query’s labeled row cannot be evaluated (see `skipped_ground_truth`).

### Metrics JSON (output — created every run)

When `--gt_json` is set, after the run the script writes e.g.:

`copy_v2_results/metrics/metrics_20260326_203316.json`

Typical fields:

| Field | Meaning |
|--------|--------|
| `torchmetrics_retrieval_map` | **mAP** from TorchMetrics `RetrievalMAP` (higher is better, max 1.0). |
| `num_labeled_queries_evaluated` | How many **labeled query detections** were fully evaluated. |
| `num_pairwise_records` | How many **(reference × query)** score rows were fed into the metric (grows with `#refs × #labeled queries`). |
| `skipped_ground_truth` | *(Optional)* List of `{ query_image, query_box_idx, reason }` for GT rows that could not be evaluated. |

This file is **separate** from the ground-truth JSON: ground truth is **input** you maintain; metrics JSON is **output** logging that run’s scores and paths.

### Command-line usage (`parse_args`)

Run the script from the **`reidentification_pipeline/`** directory so imports resolve (`dinov2_extractor` lives next to the demo). Use **absolute paths** for inputs/outputs if you run from elsewhere.

**Positional (required, first argument):**

| Argument | Meaning |
|----------|--------|
| `text_prompt` | SAM3 text prompt (quoted if it has spaces), e.g. `"yellow ball"`. No `--` flag. |

**Optional flags:**

| Long | Short | What it sets |
|------|-------|----------------|
| `--ref_image_path` | `-r` | Single **reference image** file (PNG/JPG). SAM3 segments this first; DINO uses its boxes as columns in the similarity matrix. |
| `--image_directory` | `-i` | **Folder** of query images (`.png`, `.jpg`, `.jpeg`). All are processed in sorted order (use `--max_query_images` to cap). |
| `--output` | `-o` | **Folder** where segmented PNGs and `metrics/` are written. Created if missing. |
| `--max_query_images` | `-max` | Process only the first *N* query images (omit for all images in the directory). |
| `--dinov2_model` | — | DINOv2 variant: `vits14`, `vitb14`, `vitl14`, `vitg14`, or `*_reg` variants (see script). |
| `--sam3_model` | — | Hugging Face model id for SAM3 (default `facebook/sam3`). |
| `--gt_json` | — | Path to a **single** ground-truth **`.json` file`** (not a directory). Omit to skip mAP and metrics JSON. |

**Help:** `python3 demo_spot_reid_v2_Copy.py -h` prints the same interface argparse built from `parse_args()`.

**Minimal run** (similarity matrices + segmented images only; no ground truth, no mAP):

```bash
cd reidentification_pipeline
python3 demo_spot_reid_v2_Copy.py "yellow ball" \
  --ref_image_path "/home/you/astr_research/reference_image/20250529_124829_Camera_-1_-1_315.png" \
  --image_directory "/home/you/astr_research/temp_folder" \
  --output "/home/you/astr_research/copy_v2_results"
```

**Full run** (same as above, plus TorchMetrics mAP and a metrics JSON under `--output/metrics/`):

```bash
cd reidentification_pipeline
python3 demo_spot_reid_v2_Copy.py "yellow ball" \
  --ref_image_path "/home/you/astr_research/reference_image/20250529_124829_Camera_-1_-1_315.png" \
  --image_directory "/home/you/astr_research/temp_folder" \
  --output "/home/you/astr_research/copy_v2_results" \
  --gt_json "/home/you/astr_research/copy_v2_results/json_results/ground_truth_mock.json"
```

Replace `/home/you/astr_research` with your clone path. If the Hugging Face SAM3 model is gated, set `HUGGINGFACE_HUB_TOKEN` in the environment before running.

---

## DINOv2 Setup Guide — Step-by-Step

This section walks you through setting up DINOv2 manually for spot-guided reidentification, from environment creation to running your first inference.

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

This installs Pillow, NumPy, tqdm, matplotlib, **torchmetrics** (for mAP when using `--gt_json`), etc. DINOv2 itself only needs PyTorch; the SAM3 + re-ID script also needs `transformers` (see comments in `requirements.txt`).

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
├── README.md                 # this file (setup + SAM3/DINO re-ID + TorchMetrics)
├── requirements.txt
├── reidentification_pipeline/
│   ├── dinov2_extractor.py
│   └── demo_spot_reid_v2_Copy.py   # SAM3 + DINO + optional mAP
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
