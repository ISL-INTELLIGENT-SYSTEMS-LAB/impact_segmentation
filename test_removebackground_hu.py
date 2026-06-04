import os
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from huggingface_hub import login
from math import log10, copysign, exp

hf_token = os.environ["HFTOKEN"]
login(hf_token)

def get_sam3_results(image,text_prompt,device):
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device) 
    # Prompt the model with text
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    sam3_results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist())[0]
    return sam3_results

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
IMG_REF = r"D:\hu_moments\IMG_2245.jpg"
IMG_PTH = r"D:\hu_moments\IMG_2246.jpg"
TEXT_PROMPT = "blue mug"
SAM_MODEL_CFG = "facebook/sam3"
y = 3

img_ref_pil = Image.open(IMG_REF).convert("RGB")
img_ref_np = np.array(img_ref_pil)   # H W 3 uint8

img_pil = Image.open(IMG_PTH).convert("RGB")
img_np = np.array(img_pil)   # H W 3 uint8

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
sam3_results_ref = get_sam3_results(img_ref_pil,TEXT_PROMPT,device)
sam3_results_img = get_sam3_results(img_pil,TEXT_PROMPT,device)

#print(sam3_results)
masks_ref  = sam3_results_ref["masks"].cpu().numpy()
masks_img  = sam3_results_img["masks"].cpu().numpy()

boxes_ref  = sam3_results_ref["boxes"].cpu().numpy()
boxes_img  = sam3_results_img["boxes"].cpu().numpy()

scores_ref = sam3_results_ref["scores"].cpu().numpy()
scores_img = sam3_results_img["scores"].cpu().numpy()
#labels = sam3_results["labels"]

"""for i in range(len(masks)):
    mask  = masks[i]
    bbox   = boxes[i]
    score = scores[i]
    label = TEXT_PROMPT#[i]   # SAM3 assigns the prompt index
    
    crop = crop_from_mask(img0_np, mask, bbox)
    plt.imshow(crop)
    plt.show()
    """
#print(masks_ref[0])
crop_ref = crop_from_mask(img_ref_np, masks_ref[0], boxes_ref[0])
crop_img = crop_from_mask(img_np, masks_img[0], boxes_img[0])

gray_ref = Image.fromarray(crop_ref)
gray_img = Image.fromarray(crop_img)

gray_ref = gray_ref.convert("L")
gray_img = gray_img.convert("L")
#print(len(gray_ref.getbands()))

_,gray_ref = cv2.threshold(np.array(gray_ref), 128, 255, cv2.THRESH_BINARY)
_,gray_img = cv2.threshold(np.array(gray_img), 128, 255, cv2.THRESH_BINARY)

#moments_ref = cv2.moments(gray_ref)
#hu_moments_ref = cv2.HuMoments(moments_ref)

#moments_img = cv2.moments(gray_img)
#hu_moments_img = cv2.HuMoments(moments_img)

"""for i in range(0,7):
   hu_moments_ref[i] = -1* copysign(1.0, hu_moments_ref[i]) * log10(abs(hu_moments_ref[i]))
"""
"""for i in range(0,7):
   hu_moments_img[i] = -1* copysign(1.0, hu_moments_img[i]) * log10(abs(hu_moments_img[i]))
"""
#print(hu_moments_ref)
#print(hu_moments_img)
distance = cv2.matchShapes(gray_ref, gray_img, cv2.CONTOURS_MATCH_I2, 0)
print(distance)

#d_ = distance / (1 + distance)
#s_shape = exp(-y * d_)

s_shape = exp(-y * distance)
print(s_shape)


