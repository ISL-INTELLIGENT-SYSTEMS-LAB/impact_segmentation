import os
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import Sam3Processor, Sam3Model
from huggingface_hub import login

hf_token = os.environ["HUGGINGFACE_HUB_TOKEN"]
login(hf_token)

def crop_from_mask(image, mask, bbox):
    x0, y0, x1, y1 = map(int, bbox)

    # crop the region of interest
    crop = image[y0:y1, x0:x1].copy()

    # crop the mask to the same region
    mask_crop = mask[y0:y1, x0:x1]

    # apply mask: zero out background
    crop[~mask_crop.astype(bool)] = 0

    return crop

   
# =============================================================================
# Load the Input Image Pair using pillow, and resize to same size used in LoFTR
# =============================================================================
IMG0_PTH = "/home/cspooner/IMPACT1/data_collection/scene_integration/placement_data_20250602_141151/20250602_141151_Camera_-1_3_225.png"
TEXT_PROMPT = "brown scrubber" 
SAM_MODEL_CFG = "facebook/sam3"
img0_pil = Image.open(IMG0_PTH).convert("RGB")
img0_np = np.array(img0_pil)   # H W 3 uint8

#img0_pil = img0_pil.resize((RESIZE_WIDTH, RESIZE_HEIGHT ))

# =============================================================================
# SAM3 Mask Generation
# =============================================================================

# select the device for computation
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model = Sam3Model.from_pretrained(SAM_MODEL_CFG).to(device)
processor = Sam3Processor.from_pretrained(SAM_MODEL_CFG)

# Load an image

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
    target_sizes=inputs.get("original_sizes").tolist())[0]

print(sam3_results)
masks  = sam3_results["masks"].cpu().numpy()
boxes  = sam3_results["boxes"].cpu().numpy()
scores = sam3_results["scores"].cpu().numpy()
#labels = sam3_results["labels"]

for i in range(len(masks)):
    mask  = masks[i]
    bbox   = boxes[i]
    score = scores[i]
    label = TEXT_PROMPT#[i]   # SAM3 assigns the prompt index
    
    crop = crop_from_mask(img0_np, mask, bbox)
    plt.imshow(crop)

    #sem = semantic_embedding(crop)
    #geom = geometric_features(crop, registry[label])
    #shape = shape_descriptor(mask)
    #pos = object_center(mask)

    #inst_id = match_or_register(label, sem, geom, shape, pos)
plt.show()

      
      
