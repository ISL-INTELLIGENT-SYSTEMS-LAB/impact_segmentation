import os
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from huggingface_hub import login
from math import log10, copysign, exp
import yaml
from pathlib import Path
from PIL import Image
from transformers import Sam3Processor, Sam3Model

hf_token = os.environ["HUGGINGFACE_HUB_TOKEN"]
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

def load_hu(crop1, crop2, y):

  gray_ref = Image.fromarray(crop1)
  gray_img = Image.fromarray(crop2)

  gray_ref = gray_ref.convert("L")
  gray_img = gray_img.convert("L")

  _,gray_ref = cv2.threshold(np.array(gray_ref), 128, 255, cv2.THRESH_BINARY)
  _,gray_img = cv2.threshold(np.array(gray_img), 128, 255, cv2.THRESH_BINARY)

  distance = cv2.matchShapes(gray_ref, gray_img, cv2.CONTOURS_MATCH_I2, 0)
  #print(distance)

  hu_sim_score = exp(-y * distance)
  return(hu_sim_score)
  
if __name__ == "__main__":
  # =============================================================================
  # Load the Input Image Pair using pillow, and resize to same size used in LoFTR
  # =============================================================================

  # Open and parse the YAML configuration file.
  with open("sceneREID_config.yaml", "r") as f:
      config = yaml.safe_load(f)

  # Assign configuration parameters to variables
  EXPERIMENT_NAME = config["EXPERIMENT_NAME"]
  IMG_REF = Path(config["IMG0_PTH"])
  IMG_PTH = Path(config["IMG1_PTH"])
  TEXT_PROMPT = config["TEXT_PROMPT"]
  SAM_MODEL_CFG = config["SAM_MODEL_ID"]
  MATCHING_NAME = config["MATCHING_NAME"]
  RESIZE_HEIGHT = config["RESIZE_HEIGHT"]
  RESIZE_WIDTH = config["RESIZE_WIDTH"]
  CREATE_SAVEDIR = config["CREATE_SAVEDIR"]
  VISUALIZE_FIG = config["VISUALIZE_FIG"]
  Y = config["Y"]
  
  if torch.cuda.is_available():
      DEVICE = torch.device('cuda')
  else:
      DEVICE = torch.device('cpu')
      
  img_ref_pil = Image.open(IMG_REF).convert("RGB")
  img_ref_np = np.array(img_ref_pil)   # H W 3 uint8

  img_pil = Image.open(IMG_PTH).convert("RGB")
  img_np = np.array(img_pil)   # H W 3 uint8

  #img0_pil = img0_pil.resize((RESIZE_WIDTH, RESIZE_HEIGHT ))

  # =============================================================================
  # SAM3 Mask Generation
  # =============================================================================

  # Load the model
  model = Sam3Model.from_pretrained(SAM_MODEL_CFG).to(DEVICE)
  processor = Sam3Processor.from_pretrained(SAM_MODEL_CFG)

  # Load an image

  # Segment using text prompt
  sam3_results_ref = get_sam3_results(img_ref_pil,TEXT_PROMPT,DEVICE)
  sam3_results_img = get_sam3_results(img_pil,TEXT_PROMPT,DEVICE)

  #print(sam3_results)
  masks_ref  = sam3_results_ref["masks"].cpu().numpy()
  masks_img  = sam3_results_img["masks"].cpu().numpy()

  boxes_ref  = sam3_results_ref["boxes"].cpu().numpy()
  boxes_img  = sam3_results_img["boxes"].cpu().numpy()

  scores_ref = sam3_results_ref["scores"].cpu().numpy()
  scores_img = sam3_results_img["scores"].cpu().numpy()
  #labels = sam3_results["labels"]

  #print(masks_ref[0])
  crop_ref = crop_from_mask(img_ref_np, masks_ref[0], boxes_ref[0])
  crop_img = crop_from_mask(img_np, masks_img[0], boxes_img[0])
  
  hu_sim_score = load_hu(crop_ref, crop_img, Y)
  print(hu_sim_score)


