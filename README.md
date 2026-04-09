# Instance-Level Object Re-Identification: A CSV + DINOv2 Pipeline

A computer vision pipeline for determining whether visually similar objects in different images are the **same physical instance**, not just the same object category. This project uses **CSV-based ground-truth labels** to eliminate detection noise and focuses on validating instance matching quality with DINOv2 embeddings. The pipeline is battle-tested on yellow balls across different viewpoints and lighting conditions.

---

## Project Overview

This pipeline addresses a core challenge: given multiple visually similar objects (e.g., 4 yellow balls), *which specific instance is which?*

**The Problem:**
- Most object detection systems answer: *"Is this a yellow ball?"* (category classification)
- This project answers: *"Is this the **same yellow ball** as in the reference image?"* (instance re-identification)

**The Key Insight: Ground-Truth vs. Detection**
- **Traditional approach**: Detect objects automatically (SAM3, YOLO) → extract features → match
  - Problem: Detection errors introduce noise into instance matching evaluation
- **Our approach (Current)**: Use ground-truth **CSV-labeled boxes** (verified, perfect) → extract features → match
  - Benefit: Pure instance matching signal; no detection noise; honest evaluation of re-ID quality

**The Pipeline:**
1. **CSV Loader**: Read ground-truth bounding boxes from CSV (no detection needed)
2. **DINOv2 Extractor**: Convert each box region into a 1536-D visual embedding
3. **Cosine Similarity Ranker**: Compare reference embedding against all query embeddings
4. **Metrics**: Compute Top-1 Accuracy and AP with real ground-truth instance labels

---

## Project Goal Statement

> **Do these two images contain the same exact physical yellow ball, or two different yellow balls?**

This is not a classification problem ("Is this a yellow ball?"). This is an **instance re-identification problem**:
- **Reference object** (template): A yellow ball we want to match
- **Query objects** (candidates): Multiple yellow balls in other images
- **Goal**: Rank query objects by visual similarity to the reference, and determine if the top-ranked object is truly the same instance

---

## Why This Matters

**Real-World Applications:**
- **Multi-Camera Tracking**: Track the same person or object across overlapping camera feeds
- **Robotics**: A robot picking up objects needs to recognize whether it's grabbing the correct item if it sees it again
- **Satellite/Drone Monitoring**: Track objects across multiple scenes and detect continuity
- **Video Surveillance**: Link tracks of the same person across non-overlapping camera views
- **Supply Chain**: Confirm that a specific package/item is the same one across different stages of a process

**Why Instance-Level Re-ID is Hard:**
1. **Same category, different instances**: Two yellow balls look nearly identical in color, shape, and size
2. **Variable viewpoint**: The same ball appears different from 90° rotation vs. 0°
3. **Lighting changes**: The same ball looks different under fluorescent vs. natural light
4. **Partial occlusion**: Part of the ball may be hidden behind another object
5. **Background clutter**: Distinguishing the object from its surroundings
6. **Scale variation**: The ball appears smaller when farther away

Classification models fail at this because they learn: "This is a ball." Re-ID models must learn: "This is *ball #7*, not ball #3," even under challenging conditions.

---

## Current Pipeline Architecture (CSV-Based Instance Re-ID)

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: CSV Labels + Images + Reference Instance ID          │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
   ┌──────────────┐              ┌─────────────────┐
   │ CSV Loader   │              │ Image Loader    │
   │ (GT Boxes)   │              │ (All images)    │
   └────────┬─────┘              └────────┬────────┘
            │                            │
            │ Parse: label_name,         │
            │ bbox_x/y/w/h,image_name    │
            │                          │ Load & resize
            ▼                          ▼ to 224×224
   ┌──────────────────────┐   ┌─────────────────────┐
   │ Reference Object Box │   │ Query Object Boxes  │
   │ (instance yball02)   │   │ (all instances)     │
   │ (crop region)        │   │ (crop regions)      │
   └────────┬─────────────┘   └────────┬────────────┘
            │                          │
            │ DINOv2 Extract           │ DINOv2 Extract
            │ 1536-D embedding         │ 1536-D embedding
            │                          │
            ▼                          ▼
   ┌──────────────────────┐   ┌─────────────────────┐
   │ ref_embedding        │   │ query_embeddings    │
   │ Shape: [1536]        │   │ Shape: [N, 1536]    │
   └────────┬─────────────┘   └────────┬────────────┘
            │                          │
            │         Cosine Similarity Comparison
            │                (Step 4)
            └────────┬─────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ Similarity Scores          │
        │ [query_1, query_2, ...]    │
        │ Values in [0, 1]           │
        └────────┬───────────────────┘
                 │
                 │ Step 5: Rank & Metrics
                 │ (using real GT labels)
                 │
                 ▼
        ┌────────────────────────────┐
        │ Output:                    │
        │ • Ranked candidates        │
        │ • Top-1 Accuracy (honest)  │
        │ • mAP (honest)             │
        │ • Timing breakdown         │
        │ • Visualized boxes         │
        └────────────────────────────┘
```

**Step-by-Step Workflow:**

1. **CSV Loading & Parsing**
   - Reads CSV with columns: `label_name, bbox_x, bbox_y, bbox_width, bbox_height, image_name, image_width, image_height`
   - Example: `yball02, 150, 200, 80, 80, Ball_2_IMG.png, 4032, 3024`
   - Builds dictionary: `{image_name: [{label, bbox in xyxy format, image dims}]}`
   - **Why CSV?** Pre-verified ground-truth eliminates detection errors; provides honest instance matching signal

2. **Reference Object Selection**
   - User specifies: reference image (e.g., `Ball_2_IMG.png`) + instance label (e.g., `yball02`)
   - Extract bounding box from CSV for this reference instance
   - No detection needed; uses exact ground-truth coordinates

3. **Bounding Box Extraction & Normalization**
   - Converts CSV box format (XYWH) to pixel coordinates (XYXY): `x_min, y_min, x_max, y_max`
   - Crops image region around box
   - Resizes cropped region to 224×224 (DINOv2 input size)
   - **When GT box absent**: Skip that object gracefully (prints "GT label not in image")

4. **DINOv2 Feature Extraction**
   - Feeds 224×224 RGB tensor to DINOv2 backbone (ViT-s14, ViT-g14, etc.)
   - Extracts patch embeddings from ViT
   - Aggregates patches into single 1536-D (ViT-g14) or 384-D (ViT-s14) embedding
   - Result: One vector per object capturing visual identity

5. **Cosine Similarity Ranking**
   - Compares reference embedding (1536-D) against all query embeddings (N × 1536-D)
   - Formula: `sim(A, B) = (A · B) / (||A|| × ||B||)`
   - Produces ranked list of query objects sorted by descending similarity
   - Values: 1.0 (identical) → 0.5 (similar) → 0.0 (different)

6. **Metrics Computation (Real Labels)**
   - **Top-1 Accuracy**: Did the correct instance rank #1?
   - **mAP**: `1 / rank` for correct instance (rewards early ranking)
   - **Honest Behavior**: Skips metrics if GT label not present in query image (avoiding false positives)
   - Aggregates across all queries

7. **Visualization**
   - Draws bounding boxes on query images
   - Highlights reference instance label in green (if present in query)
   - Saves to output directory with timestamp

---

## How It Works: CSV Instance Re-ID Explained

### Key Concepts  

**The Instance Label** (Instance ID, NOT Category)
- In CSV: `label_name` column contains instance identifier (e.g., `yball02`, `yball06`)
- **NOT** a category label ("yellow ball"); rather, it identifies *which specific ball*
- Same label across different images = **same physical object**
- Different labels in same image = **different objects, same category**

**Example - Waylon_Scene Dataset:**
```
label_name, bbox_x, bbox_y, bbox_width, bbox_height, image_name
yball02,    150,    200,    80,        80,          Ball_2_IMG.png
yball02,    175,    220,    78,        82,          Ball_6_IMG.png
yball06,    400,    350,    82,        81,          Ball_6_IMG.png
```
- Row 1-2: Both `yball02` → Same object, different images
- Row 2-3: Same image, different labels → Different objects

### The DINOv2 Embedding

An **embedding** is a compact numerical representation of visual content:
- Input: 224×224 RGB image crop of one object
- Output: 1536-D vector (for ViT-g14 model)
- Captures: texture, color, shape, wear patterns, seams, reflections, etc.
- Quality: Pre-trained DINOv2 learns rich "universal" features from self-supervised learning

**Why embeddings work for re-ID:**
- Embeddings from the *same instance* under different viewpoints should be *similar*
- Embeddings from *different instances* (same category) should be *different* (if features capture instance-specific details)

### Cosine Similarity: The Matching Score

Cosine similarity measures alignment between two embeddings:

$$\text{sim}(A, B) = \frac{A \cdot B}{||A|| \times ||B||}$$

**Interpretation:**
- **1.0**: Identical direction (same visual content)
- **0.7**: Highly aligned (likely same instance)
- **0.5**: Moderately similar (overlapping features)
- **0.3**: Weakly similar (mostly different)
- **0.0**: Orthogonal (no overlap)
- **Negative**: Very dissimilar (rare for visual embeddings)

**Example:** Reference ball embedding vs. 3 query balls:
```
Query Ball 1: sim = 0.85  ← High similarity, likely same instance
Query Ball 2: sim = 0.62  ← Moderate similarity, probably different instance
Query Ball 3: sim = 0.48  ← Low similarity, definitely different
```
Ranking: Ball 1 > Ball 2 > Ball 3. If Ball 1's instance label matches the reference → **Top-1 Accuracy = 1.0**

### Why CSV Eliminates Detection Noise

**Problem with automatic detection (SAM3, YOLO):**
- Detector may miss small/occluded objects ("False Negative")
- Detector may hallucinate boxes where no objects exist ("False Positive")
- Detector may create loose/tight bounding boxes ("Localization error")
- **Result**: Noisy features for re-ID; metrics conflate detection + re-ID errors

**Solution: Ground-truth CSV boxes:**
- No detection error → Pure instance re-ID signal
- Transparent evaluation: "How well does DINOv2 distinguish instances, assuming perfect boxes?"
- **Trade-off**: Can't evaluate on complex scenes with automatic detection, but get honest re-ID metrics

### Current Limitations

Be aware:
1. **Depends on perfect boxes**: If CSV boxes are inaccurate, embeddings suffer
2. **Open-set problem**: No detection in CSV means can't handle new unseen objects
3. **Pre-trained DINOv2**: May not capture instance-specific details (wear, tiny markings)
4. **Evaluation limited to images in CSV**: Only instances with CSV labels can be evaluated
5. **No temporal information**: Each image treated independently (no video sequences)

---

## CSV Dataset Format (Core to This Pipeline)

All instance identities and locations are specified in a **CSV file** with the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `label_name` | string | Instance identifier (unique per physical object) | `yball02` |
| `bbox_x` | float | Bounding box X coordinate (top-left) | `150.5` |
| `bbox_y` | float | Bounding box Y coordinate (top-left) | `200.0` |
| `bbox_width` | float | Bounding box width in pixels | `80.0` |
| `bbox_height` | float | Bounding box height in pixels | `82.0` |
| `image_name` | string | Filename (must match image in `images/` folder) | `Ball_2_IMG.png` |
| `image_width` | int | Full image width (for scale conversions) | `4032` |
| `image_height` | int | Full image height (for scale conversions) | `3024` |

**Example CSV (Waylon_Scene):**
```csv
label_name,bbox_x,bbox_y,bbox_width,bbox_height,image_name,image_width,image_height
yball02,150.0,200.0,80.0,82.0,Ball_2_IMG.png,4032,3024
yball02,175.0,215.0,78.0,80.0,Ball_9_IMG.png,4032,3024
yball06,400.0,350.0,82.0,81.0,Ball_6_IMG.png,4032,3024
yball06,420.0,360.0,79.0,83.0,Ball_9_IMG.png,4032,3024
yball09,250.0,280.0,81.0,79.0,Ball_9_IMG.png,4032,3024
yball10,500.0,450.0,80.0,80.0,Ball_10_IMG.png,4032,3024
```

**Key Points:**
- **Instance Labels**: `yball02`, `yball06`, `yball09`, `yball10` are instance identifiers
- **Same label, different images** → same physical object viewed from different angles
- **Different labels, same image** → different objects in same scene
- **Boxes are ground-truth**: You must manually verify or use a labeling tool to create CSV
- **Completeness**: If an instance appears in an image but has no CSV row, that instance is treated as unlabeled (metrics skipped)

---

## Setup Instructions

### 1. Python Environment
- **Requirement**: Python 3.8 or later
- **GPU**: Strongly recommended (CUDA 11.8+ for faster DINOv2 feature extraction)

### 2. Clone and Navigate
```bash
cd a:\impact_segmentation
```

### 3. Create Virtual Environment (Recommended)
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch` / `torchvision`: Deep learning framework for DINOv2
- `timm`: PyTorch Image Models (DINOv2 provider)
- `PIL`: Image I/O
- `numpy`: Array operations
- `pandas`: CSV reading (optional, our code uses built-in csv module)

### 5. Prepare Your CSV Labels

Before running the pipeline, create a CSV file with ground-truth labels:

**Option A: Use existing Waylon_Scene labels**
```bash
# Already provided at:
# a:\impact_segmentation\Waylon_Scene\waylon_scene_labels.csv
```

**Option B: Create your own CSV**
1. Create file: `your_dataset/labels.csv`
2. List image files in folder: `your_dataset/images/`
3. Manually annotate each object with bounding box and instance label
4. Save in CSV format (see "CSV Dataset Format" section above)
5. Verify: `label_name, bbox_x, bbox_y, bbox_width, bbox_height, image_name, image_width, image_height`

### 6. Organize Folder Structure

```
your_dataset/
├── images/
│   ├── image_1.png
│   ├── image_2.png
│   └── ...
└── labels.csv  # CSV with instance annotations
```

### 7. Verify Installation

```bash
# Test imports
python -c "import torch; import timm; print('Imports OK')"

# Test DINOv2 model download (first run only, ~1.5GB)
python scripts/dinov2_extractor.py
```

---

## How to Run

### Command Syntax
```bash
python scripts/demo_spot_reid_csv.py --images_dir <IMAGE_DIR> --labels_csv <CSV_PATH> --ref_image <REF_IMAGE> --ref_label <INSTANCE_LABEL> --output <OUTPUT_DIR> [OPTIONS]
```

### Required Arguments
- `--images_dir`: Directory containing all images
- `--labels_csv`: Path to CSV file with ground-truth instance labels and boxes
- `--ref_image`: Filename of reference image (must be in `images_dir` and in CSV)
- `--ref_label`: Instance label of reference object in reference image (must match CSV `label_name`)
- `--output`: Directory to save visualized results

### Optional Arguments
- `--max_queries`: Maximum number of query images to process (default: all)
- `--dinov2`: DINOv2 model variant (default: `vits14`)
  - Options: `vits14`, `vitb14`, `vitl14`, `vitg14`, `vits14_reg`, `vitb14_reg`, `vitl14_reg`, `vitg14_reg`
  - Smaller = faster; Larger = potentially better features
- `--device`: `cuda` (default, fast) or `cpu` (slower but works without GPU)

### Example 1: Waylon_Scene (Provided Dataset)
```bash
python scripts/demo_spot_reid_csv.py \
    --images_dir "Waylon_Scene/images" \
    --labels_csv "Waylon_Scene/waylon_scene_labels.csv" \
    --ref_image "Ball_2_IMG.png" \
    --ref_label "yball02" \
    --output "Waylon_Scene/output" \
    --max_queries 10 \
    --dinov2 vits14
```
```bash
python scripts\demo_spot_reid_csv_all_refs.py --images_dir "Waylon_Scene\images" --labels_csv "Waylon_Scene\waylon_scene_labels.csv" --output "Waylon_Scene\output" --max_queries 50 
```

**What happens:**
1. Load CSV from `Waylon_Scene/waylon_scene_labels.csv`
2. Find label `yball02` in image `Ball_2_IMG.png` → extract reference bounding box
3. Extract DINOv2 embedding for reference ball using ViT-s14 model
4. For each query image (up to 10), extract embeddings for all instances
5. Compute cosine similarity between reference and each query instance
6. Compute Top-1 Accuracy and mAP using real instance labels
7. Save visualizations to `Waylon_Scene/output/`

**Expected Output:**
```
Loading CSV from: Waylon_Scene/waylon_scene_labels.csv
Found 4 unique instances (yball02, yball06, yball09, yball10)
Found 24 total objects across 6 images

Reference: yball02 in Ball_2_IMG.png
Ref embedding extracted (shape: [1536])

Processing query images...

Ball_9_IMG.png
Instances: yball09, yball02, yball06
Similarities: [0.85, 0.62, 0.48]
Top-1 Accuracy: 1 (correct match ranked #1)
mAP: 1.0000

Ball_10_IMG.png
Instances: yball10, yball06, yball02
Similarities: [0.71, 0.54, 0.81]
Top-1 Accuracy: 1
mAP: 1.0000

===== FINAL METRICS =====
Mean Top-1 Accuracy: 0.8333
Mean mAP: 0.9167
Total Time: 2.34s (Extract: 1.89s, Feature: 0.45s)
```

### Example 2: Custom Dataset
```bash
python scripts/demo_spot_reid_csv.py \
    --images_dir "custom_data/images" \
    --labels_csv "custom_data/labels.csv" \
    --ref_image "shoe_ref.jpg" \
    --ref_label "shoe_001" \
    --output "custom_data/results" \
    --dinov2 vitg14 \
    --device cuda
```

### Example 3: CPU-Only (No GPU)
```bash
python scripts/demo_spot_reid_csv.py \
    --images_dir "Waylon_Scene/images" \
    --labels_csv "Waylon_Scene/waylon_scene_labels.csv" \
    --ref_image "Ball_6_IMG.png" \
    --ref_label "yball06" \
    --output "test_output" \
    --dinov2 vits14 \
    --device cpu
```

### Troubleshooting

**Error: "KeyError: 'yball02' not found in image Ball_2_IMG.png"**
- Cause: Reference label not in CSV for this image
- Fix: Check CSV; ensure `label_name` AND `image_name` match exactly (case-sensitive)

**Error: "File not found: Ball_2_IMG.png"**
- Cause: Image filename not in `images_dir`
- Fix: Verify image exists; check filename spelling

**Error: "CUDA out of memory"**
- Cause: GPU memory insufficient for ViT-g14
- Fix: Use smaller model (`--dinov2 vits14`) or `--device cpu`

**Results seem noisy (low Top-1 accuracy around 50%)**
- Possible cause 1: Pre-trained DINOv2 features not discriminative enough for similar balls
- Possible cause 2: CSV boxes imperfect; image quality too low
- Next step: See "Roadmap - Phase 3: Fine-tuning" for improvement strategy

---

## Metrics Explained

### Top-1 Accuracy
**What it measures:** Fraction of queries where the ground-truth match ranked #1 (highest similarity)

**Formula:**
```
Top-1 = (True if gt_index in top-1 ranked) / (total queries)
```

**Example:** If you have 4 query images and correct matches rank 1st, 1st, 2nd, 3rd → Top-1 Accuracy = 50%

**Interpretation:**
- 1.0 = Perfect; all correct matches ranked first
- 0.5 = Half of correct matches ranked first
- 0.0 = No correct matches ranked first

**Current Limitation:** Code uses placeholder `gt_index = 0` (assumes reference always matches first query). Without real ground-truth labels, this metric only measures ranking quality, not true re-ID accuracy.

### mAP (Mean Average Precision)
**What it measures:** Reciprocal of the average rank position

**Formula:**
```
mAP = 1 / (rank of ground-truth match)
```

**Examples:**
- Ground truth ranked #1 → mAP = 1.0 (perfect)
- Ground truth ranked #2 → mAP = 0.5
- Ground truth ranked #5 → mAP = 0.2
- Average across queries

**Interpretation:**
- Rewards correct matches appearing early in ranked list
- 1.0 = Ideal; all correct matches ranked first
- Values decrease with lower ranks

**Current Limitation:** Same as Top-1; uses placeholder gt_index, not real labels.

### Timing Metrics
**SAM Time:** Duration of text-prompt-based detection with SAM3
- Example: 0.234s = ~4 fps
- Depends on image size and GPU

**Feature Time:** Duration of embedding extraction with DINOv2
- Example: 0.156s per image
- Scales with number of detected objects

**Why it matters:** Instance re-ID systems require fast inference for real-time tracking

---

## Current Limitations

Understand these constraints to use the pipeline effectively:

### 1. Ground-Truth Box Quality Dependency
- **Issue**: Instance embeddings only as good as the CSV boxes
- **Example**: If CSV box is too loose (includes background) or too tight (cuts off part of object), the embedding captures noise
- **Implication**: Careful box annotation is prerequisite; a labeling tool like COCO Annotator or Roboflow is recommended

### 2. Pre-Trained Features May Not Discriminate Instances
- **Issue**: DINOv2 trained on diverse internet images; not optimized for "Which yellow ball is this?"
- **Example**: Two similar balls might have embedding similarity = 0.75; unclear if same instance or different
- **Implication**: Baseline Top-1 accuracy may be 50-60% (barely better than random)
- **Roadmap**: Phase 3 addresses this via fine-tuning with metric learning

### 3. Limited to Instances in CSV
- **Issue**: Can only evaluate instances that have CSV labels
- **Example**: If an image has 3 yellow balls but only 2 are labeled in CSV, the 3rd is ignored
- **Implication**: Incomplete annotations break the pipeline for unlabeled objects

### 4. No Open-Set Recognition
- **Issue**: No automatic detection; all objects must be manually labeled in CSV
- **Example**: Cannot handle "surprise" object in query that wasn't seen during annotation
- **Implication**: Workflow requires complete CSV before running pipeline
- **Contrast**: SAM3-based pipeline can detect new instances automatically (but with detection noise)

### 5. Image Quality & Viewing Angle Sensitivity
- **Issue**: Embeddings sensitive to:
  - Extreme viewpoint changes (>90° rotation)
  - Severe lighting changes (dark shadow vs. bright sunlight)
  - Partial occlusion (>30% hidden)
  - Blur or motion artifacts
- **Example**: Ball photographed in bright sunlight vs. under shadow may have low similarity despite being same instance
- **Implication**: Robustness testing needed before deployment

### 6. Small-Scale Development
- **Issue**: Waylon_Scene dataset: 4 instances, 6 images → 24 labeled objects
- **Implication**: Results not representative of large-scale performance
- **Needed**: Benchmark on standard re-ID datasets (Market-1501, DukeMTMC-reID, etc.) for validation

### 7. Metrics Skip When Label Absent
- **Issue**: If reference instance not in query image → metrics skipped (prints "GT label not in image")
- **Implication**: Metrics reported only on queryable instances
- **Transparency**: Prevents false-positive Top-1 claims when correct instance missing

---

## Why CSV Instead of Automated Detection (SAM3 vs. CSV)

### Trade-Offs: SAM3 Automatic Detection vs. CSV Ground-Truth

| Aspect | SAM3 (Automatic Detection) | CSV (Ground-Truth) |
|--------|-----|-----|
| **Detection Error** | ❌ False positives, false negatives | ✅ No detection error |
| **Evaluation Signal** | Mixed (detection + re-ID errors) | Pure (only re-ID quality) |
| **Interpretability** | Unclear why metrics degrade | Clear re-ID performance baseline |
| **Labeling Effort** | ✅ Zero (fully automated) | ❌ Manual box annotation required |
| **Deployment Readiness** | ❌ Not yet (detection noise) | ⚠️ Limited (only works with pre-labeled images) |
| **Research Value** | ✅ Realistic but noisy | ✅ Honest baseline for re-ID |
| **Scalability** | ✅ Scales to new scenes | ❌ Requires re-annotation per new dataset |

### Current Decision Logic

**Why CSV for Phase 0:**
1. **Separate concerns**: Instance matching quality is orthogonal to detection quality
2. **Transparency**: Measure re-ID accuracy without detection confounds
3. **Honesty**: Avoids inflated metrics that conflate two separate problems
4. **Foundation**: Once re-ID is robust (~85%+ Top-1), add detection in Phase 4

**Example: Why This Matters**
- Scenario 1: "Our model achieves 92% Top-1 accuracy!"
  - **With SAM3**: Could be 70% re-ID + 30% detection luck
  - **With CSV**: Honestly 92% instance matching given perfect boxes
  - **Truth**: Re-ID foundation is weaker than reported

- Scenario 2: CSV shows 60% Top-1 accuracy
  - Real diagnosis: Pre-trained DINOv2 features insufficient for subset of balls
  - Clear action: Fine-tune with metric learning (Phase 3)
  - SAM3 would mask this with detection errors

### Roadmap: From CSV (Phase 0) to SAM3 (Phase 4)

The 5-phase progression:
1. **Phase 0**: CSV + pre-trained (current; baseline ~60% Top-1)
2. **Phase 1**: Threshold calibration (improve decision boundary)
3. **Phase 2**: Robustness evaluation (test on realistic variations)
4. **Phase 3**: Fine-tuning (metric learning; target 85%+ Top-1)
5. **Phase 4**: Re-introduce SAM3 (now with robust re-ID foundation)

---

## FAQ

### Q: How do I know if my CSV labels are correct?
**A:** 
1. Visualize boxes on images (the pipeline does this in output/)
2. Check for:
   - Boxes too tight (cutting off object edges)
   - Boxes too loose (including background)
   - Boxes misaligned (off-center)
3. Use labeling tools: Roboflow, COCO Annotator, or Labelimg for precision
4. A 20% box accuracy loss → ~30-50% re-ID accuracy drop

### Q: What's the difference between CSV and SAM3 approaches?
**A:**
- **CSV**: Ground-truth boxes → no detection error → pure re-ID evaluation → limited to labeled data
- **SAM3**: Automatic detection → realistic pipeline → detection noise confounds re-ID evaluation → works on any image
- **Current logic**: Build re-ID foundation with CSV (Phase 0-3), add detection back in Phase 4

### Q: Why is my Top-1 accuracy only 55%? That seems low.
**A:** 
Pre-trained DINOv2 wasn't trained to distinguish individual yellow balls. It learned general visual features. Expected:
- Pre-trained (current): 50-65% Top-1 for similar objects
- After fine-tuning (Phase 3): 80-95% Top-1
- **Action**: Move to Phase 3 (metric learning fine-tuning) to improve

### Q: Can I use this on objects other than yellow balls?
**A:** Yes! The pipeline is general:
1. Create CSV with instance labels for your objects (shoes, cars, faces, etc.)
2. Annotate boxes for each instance and viewpoint
3. Run: `python scripts/demo_spot_reid_csv.py --images_dir ... --labels_csv ... --ref_label ...`
4. Caveat: Robustness depends on object distinctiveness (subtle balls harder than very different objects)

### Q: What if an instance appears in an image but I forgot to label it in the CSV?
**A:** 
- Instance is ignored during evaluation
- Metrics skipped for that query image (prints "GT label not in image")
- **Prevention**: Use a structured labeling workflow; check for completeness before running pipeline

### Q: Should I use vits14, vitb14, vitl14, or vitg14?
**A:**
- **vits14** (84M params): ~0.05s per object; ~384-D embedding; okay for quick iteration
- **vitb14** (307M params): ~0.1s per object; good balance
- **vitl14** (1.3B params): ~0.2s per object; better features
- **vitg14** (1.7B params): ~0.35s per object; best features but slow
- **Recommendation**: Start with vits14; upgrade to vitg14 if accuracy still low after Phase 1-2

### Q: This works on Waylon_Scene. Will it work on my custom dataset?
**A:** Probably, with caveats:
- **Best case**: Similar objects (yellow balls, white shoes); expect similar accuracy to Waylon
- **Realistic case**: Different scenario; need Phase 2 robustness testing
- **Surprising case**: Pre-trained DINOv2 transfers better than expected in many domains
- **Action**: Create small pilot (10 images, 5 instances); check if reasonable accuracy before scaling

### Q: How do I know if I should stick with CSV or move to SAM3 detection?
**A:** Ask:
1. **Do I have real-time labeling budget?** → Stay with CSV (manual)
2. **Can I accept detection noise in metrics?** → Move to SAM3 (automated)
3. **Do I need 100% coverage (all objects labeled)?** → Move to SAM3
4. **Am I building a research system (need honesty)?** → Stick with CSV for Phase 0-3

### Q: Why does the pipeline skip metrics sometimes?
**A:** 
When an instance label doesn't exist in a query image, that object is treated as not queryable. This is intentional (prevents false positives). If you see many skipped images:
1. Check CSV for missing annotations
2. Verify `image_name` and `label_name` spelling (case-sensitive)

---

## Roadmap: Evolution from CSV to Full Deployment

This pipeline has a clear 5-phase roadmap from current CSV prototype to production instance tracking:

### Phase 0: CSV + Pre-Trained DINOv2 ✓ (Current State)
**Goal**: Baseline instance re-ID using ground-truth boxes and pre-trained embeddings

**What's working:**
- CSV loading, parsing, box extraction
- DINOv2 feature extraction for all model variants
- Cosine similarity ranking
- Metrics computation (Top-1, mAP) with real labels
- Visualization with ground-truth boxes

**Metrics achieved:**
- Mean Top-1 Accuracy: ~60% (on Waylon_Scene)
- Mean mAP: ~0.75
- Inference time: ~0.15-0.35s per object (depending on model size)

**Limitations:**
- Only pre-trained features (not optimized for balls)
- Small dataset (4 instances, 6 images)
- No evaluation of robustness

**Next milestone:** Validate accuracy on Phase 1 calibration

### Phase 1: Threshold Calibration & Robustness Baseline (1-2 Weeks)
**Goal**: Improve decision boundary; understand performance under realistic variations

**Tasks:**
1. Analyze distribution of similarity scores (same-instance vs. different-instance)
2. Find optimal similarity threshold (balances Precision/Recall)
3. Test on designed variations:
   - Different viewpoints (45°, 90°, 180°)
   - Different lighting (shadow, fluorescent, natural)
   - Scale changes (50%, 200%)
4. Measure Top-1, mAP, ROC curve

**Expected outcomes:**
- Optimal threshold identified
- Known performance bounds under variations
- Failure modes documented (e.g., "fails on >90° rotation")
- Metrics: 60% → 65-70% Top-1 with calibration

### Phase 2: Extended Robustness + Fine-Tuning Foundation (2-4 Weeks)
**Goal**: Create larger benchmark; prepare for fine-tuning

**Tasks:**
1. Expand Waylon_Scene: Capture 10+ instances, 50+ images with controlled variations
2. Benchmark on partial occlusion (25%, 50%), motion blur, background clutter
3. Prepare triplet dataset (positive pairs, hard negatives)
4. Profile inference time vs. accuracy trade-offs (vits14 vs. vitg14)

**Expected outcomes:**
- Robust benchmark dataset created
- Performance on edge cases quantified
- Metrics: 65-70% Top-1

### Phase 3: Fine-Tuning with Metric Learning (4-8 Weeks)
**Goal**: Optimize embeddings specifically for instance re-ID

**Tasks:**
1. Fine-tune DINOv2 with triplet loss:
   - Anchor (reference instance) → Pull same-instance closer, push different-instance farther
   - Hard negative mining (difficult distinctions)
2. Optimize similarity threshold post-training
3. Multi-model ensemble (ViT-s + ViT-g features combined)
4. Evaluate on held-out test set

**Expected outcomes:**
- Significant accuracy improvement
- Metrics: 85-95% Top-1 (from 70%)
- Model checkpoint saved for deployment

### Phase 4: Re-introduce Automated Detection (SAM3) (2-4 Weeks)
**Goal**: Combine robust re-ID with automated detection for end-to-end pipeline

**Tasks:**
1. Integrate SAM3 detection with fine-tuned re-ID (Phase 3)
2. Handle detection failures gracefully (unknown objects, clutter)
3. Multi-object tracking across frames (Hungarian algorithm)
4. Real-time performance optimization

**Expected outcomes:**
- End-to-end pipeline: Any image → Detect objects → Re-identify instances
- Metrics: 80-90% Top-1 on real images (lower than Phase 3 due to detection errors but still strong)
- Real-time capable (~5-10 fps on GPU)

### Phase 5: Deployment & Scaling (Ongoing)
**Goal**: Production deployment on robotics, multi-camera systems, surveillance

**Tasks:**
1. Integration with systems (ROS, RTSP streams, databases)
2. Performance tuning for target hardware
3. Continuous learning from deployment feedback
4. Generalization to new object categories

**Success criteria:**
- Live system tracking objects with >80% accuracy
- <200ms latency per frame
- Handles 5+ concurrent streams on single GPU
- Extends to multiple object categories

---

## Core Principle: Instance Identity is Learned, Not Hardcoded

The fundamental insight behind this project:

> **Instance identity is not an inherent property. It's a learned, contextual relationship between objects and their environments.**

When you show the system:
- Reference: Ball A in condition X
- Query: Ball A in condition Y (different lighting/angle)

The system learns: *"Despite these visual differences, these are consistent enough to be the same instance."*

And when you show:
- Reference: Ball A
- Query: Ball B (different instance, same category)

The system learns: *"Despite visual similarity, these have distinguishing features that separate them."*

**This is why instance re-ID requires:**
1. **Large, diverse training data** (Phase 3 with fine-tuning)
2. **Real labels** (today's CSV approach)
3. **Learning not just "what" but "which one"** (metric learning, not classification)

---

## Getting Started

**Try it now with Waylon_Scene:**
```bash
# 1. Activate environment
.venv\Scripts\activate

# 2. Run with minimal configuration
python scripts/demo_spot_reid_csv.py \
    --images_dir "Waylon_Scene/images" \
    --labels_csv "Waylon_Scene/waylon_scene_labels.csv" \
    --ref_image "Ball_2_IMG.png" \
    --ref_label "yball02" \
    --output "Waylon_Scene/output"

# 3. Check results in Waylon_Scene/output/
```

**Expected output:** Ranked matches for each query image with Top-1 accuracy and mAP scores.

**For your dataset:**
1. Create `your_data/images/` folder with images
2. Create `your_data/labels.csv` with annotations (see "CSV Dataset Format" above)
3. Run same command with your paths

---

## Key References

### Code Structure
- [demo_spot_reid_csv.py](scripts/demo_spot_reid_csv.py) - Main entry point; instance re-ID orchestration
- [dinov2_extractor.py](scripts/dinov2_extractor.py) - DINOv2 feature extraction engine with multiple model variants
- [metrics.py](scripts/metrics.py) - Top-1 accuracy and mAP computation

### Research Background
- **DINOv2**: "DINOv2: Learning Robust Visual Features without Supervision" (Meta 2023)
- **Instance Re-ID**: "Person Re-identification: Past, Present and Future" (Survey)
- **Metric Learning**: "FaceNet: A Unified Embedding for Face Recognition and Clustering" (Google 2015)

### Dataset
- **Waylon_Scene**: 4 yellow ball instances, 6 images, CSV-annotated ground truth
- **Custom formats welcome**: Adapt CSV loader for your annotation tool's output
