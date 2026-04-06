"""
SAM3/CLIP with LoFTR Scene Matching

This script will run the CLIP re-identification algorithm on all images in a given
directory after an initial image is used as a reference. A list of objects of interest
in the reference image will be identified and then checked for inclusiveness in the
remain images.
 
The configuration constants (such as file paths, thresholds, and image sizes) are
loaded from an external YAML file. - eventually will be phased out in favor of argparse

*NOTE:  LofTR works on grayscale images only

Before running, ensure that:
  - A 'clipLoFTR_config.yaml' file with the required settings exists in the same directory.
  - The necessary modules and dependencies are installed. Those should be identified in the
    requirements.txt file.
    
    pip install opencv-python numpy matplotlib torch torchvision pyyaml shapely
    pip install kornia kornia-moons transformers pycocotools huggingface_hub
    
"""

import os
import cv2
import torch
import yaml
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from typing import Tuple
import kornia as K
import kornia.feature as KF
import pycocotools.mask as mask_util
from matplotlib.patches import ConnectionPatch

# Import custom modules for transforms, plotting, and models
from kornia_moons.viz import draw_LAF_matches
from pathlib import Path
from shapely.geometry import Point, Polygon
from torchvision.ops import box_convert
#from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

#from sam3.model_builder import build_sam3_image_model
#from sam3.model.sam3_image_processor import Sam3Processor
from transformers import Sam3Processor, Sam3Model

from huggingface_hub import login

hf_token = os.environ["HUGGINGFACE_HUB_TOKEN"]
login(hf_token)


# =============================================================================
# Helper Functions
# =============================================================================
from collections import defaultdict

class InstanceRegistry:
    def __init__(self, threshold=0.32):
        self.instances = {}  # id ? {label, embedding}
        self.threshold = threshold
        self.next_id = 0

    def register(self, label, embedding):
        inst_id = f"{label}_{self.next_id}"
        self.instances[inst_id] = {
            "label": label,
            "embedding": embedding
        }
        self.next_id += 1
        return inst_id

    def match(self, label, embedding):
        best_id = None
        best_sim = -1

        for inst_id, data in self.instances.items():
            if data["label"] != label:
                continue

            sim = cosine_similarity(embedding, data["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_id = inst_id

        if best_sim >= self.threshold:
            return best_id

        return None
        
def identify_or_register(registry, label, embedding):
    match_id = registry.match(label, embedding)
    if match_id is not None:
        return match_id
    return registry.register(label, embedding)

def assign_instance_id(label):
    idx = instance_counters[label]
    instance_counters[label] += 1
    return f"{label}_{idx}"


def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image
    
import numpy as np

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def match_instance(
    new_sem,                 # CLIP/DINO embedding
    new_geom,                # LoFTR/SuperGlue match score
    new_shape,               # shape descriptor vector
    new_pos,                 # (x_center, y_center) normalized
    registry,                # dict: id ? feature bundle
    label,
    w_sem=0.55,
    w_geom=0.25,
    w_shape=0.15,
    w_pos=0.05,
    threshold=0.52
):
    best_id = None
    best_score = -1

    for inst_id, data in registry.items():
        if data["label"] != label:
            continue

        s_sem = cosine(new_sem, data["sem"])
        s_geom = new_geom * data["geom"]          # geometric consistency
        s_shape = cosine(new_shape, data["shape"])
        s_pos = 1.0 - np.linalg.norm(new_pos - data["pos"])

        score = (
            w_sem * s_sem +
            w_geom * s_geom +
            w_shape * s_shape +
            w_pos * s_pos
        )

        if score > best_score:
            best_score = score
            best_id = inst_id

    if best_score >= threshold:
        return best_id

    return None



    
# =============================================================================
# Load Configuration from YAML File
# =============================================================================

# Open and parse the YAML configuration file.
with open("loftr_config2.yaml", "r") as f:
    config = yaml.safe_load(f)

# Assign configuration parameters to variables
EXPERIMENT_NAME = config["EXPERIMENT_NAME"]
IMG0_PTH = Path(config["IMG0_PTH"])
IMG1_PTH = Path(config["IMG1_PTH"])
TEXT_PROMPT = config["TEXT_PROMPT"]
LOFTR_IMAGE_TYPE = config["LOFTR_IMAGE_TYPE"]
SAM_MODEL_CFG = config["SAM_MODEL_ID"]
DISPLAY_POINTS_INSIDE = config["DISPLAY_POINTS_INSIDE"]
LOFTR_MATCHING_NAME = config["LOFTR_MATCHING_NAME"]
RESIZE_HEIGHT = config["RESIZE_HEIGHT"]
RESIZE_WIDTH = config["RESIZE_WIDTH"]

device = "cuda" if torch.cuda.is_available() else "cpu"
    
# Create an output directory with a unique name based on the 
# experiment name as well as the current date and time.
now = datetime.now()
date = now.strftime("%Y%m%d")
time = now.strftime("%H%M")
OUTPUT_DIR = os.path.join(EXPERIMENT_NAME, date, time)
os.makedirs(OUTPUT_DIR, exist_ok=True)




'''
# =============================================================================
# Load and Process the Input Image Pair using kornia
# =============================================================================

# Read the first image and resize
img0 = K.io.load_image(IMG0_PTH, K.io.ImageLoadType.RGB32)[None, ...]
img0 = K.geometry.resize(img0, (RESIZE_HEIGHT, RESIZE_WIDTH), antialias=True)

# Read the second image and resize
img1 = K.io.load_image(IMG1_PTH, K.io.ImageLoadType.RGB32)[None, ...]
img1 = K.geometry.resize(img1, (RESIZE_HEIGHT, RESIZE_WIDTH), antialias=True)


# =============================================================================
# Run LoFTR Inference and Save the Matching Figure
# =============================================================================

matcher = KF.LoFTR(pretrained=LOFTR_IMAGE_TYPE)

input_dict = {
    "image0": K.color.rgb_to_grayscale(img0),  # LofTR works on grayscale images only
    "image1": K.color.rgb_to_grayscale(img1),
}

with torch.inference_mode():
    correspondences = matcher(input_dict)

mkpts0 = correspondences["keypoints0"].cpu().numpy()
mkpts1 = correspondences["keypoints1"].cpu().numpy()
mconf = correspondences["confidence"].cpu().numpy()
'''
# =============================================================================
# Load the Input Image Pair using pillow, and resize to same size used in LoFTR
# =============================================================================

img0_pil = Image.open(IMG0_PTH).convert("RGB")
img0_pil = img0_pil.resize((RESIZE_WIDTH, RESIZE_HEIGHT ))

img1_pil = Image.open(IMG1_PTH).convert("RGB")
img1_pil = img1_pil.resize((RESIZE_WIDTH, RESIZE_HEIGHT ))

#loftrmatches = draw_matches(img0_pil, img1_pil, mkpts0, mkpts1, mconf)

#matching_name = f"ALL_{LOFTR_MATCHING_NAME}_whole_image.png"

#plt.savefig(os.path.join(OUTPUT_DIR, matching_name), dpi=300, bbox_inches="tight")
#plt.show()
#plt.clf()

# =============================================================================
# SAM3 Mask Generation
# =============================================================================
# Load the model



#model = Sam3Model.from_pretrained(SAM_MODEL_CFG).to(device)
#processor = Sam3Processor.from_pretrained(SAM_MODEL_CFG)
model = Sam3Model.from_pretrained(SAM_MODEL_CFG).to(device)
processor = Sam3Processor.from_pretrained(SAM_MODEL_CFG)

# Load an image

#img0_np = np.array(img0_pil.convert("RGB"))
#inference_state = processor.set_image(img0)

# select the device for computation
device = "cuda" if torch.cuda.is_available() else "cpu"

# Segment using text prompt
inputs = processor(images=img0_pil, text=TEXT_PROMPT, return_tensors="pt").to(device) 
# Prompt the model with text
with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
sam3_results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]
print(sam3_results)


# =============================================================================
# Create a label map 
# =============================================================================

# semantic label ? next instance index
'''instance_counters = defaultdict(int)

for mask, label in sam3_results:
    unique_id = assign_instance_id(label)
    print(unique_id)
    
    
overlay = overlay_masks(img0_pil, sam3_results["masks"])
plt.imshow(overlay)
plt.show()'''

for mask in sam3_results['masks']:
    label = TEXT_PROMPT
    crop = apply_mask(image, mask)
    embedding = clip_model.encode_image(preprocess(crop))

    inst_id = identify_or_register(registry, label, embedding)
    print("Detected:", inst_id)
