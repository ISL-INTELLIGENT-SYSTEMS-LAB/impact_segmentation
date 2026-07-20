import os
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPVisionModel, CLIPImageProcessor, Sam3Processor, Sam3Model
from PIL import Image
from huggingface_hub import login
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
from math import log10, copysign, exp
from datetime import datetime
from scipy.optimize import linear_sum_assignment
import cv2
import logging

#### these suppress some warnings that arent important ####
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

class PartBasedEmbedding(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", num_parts=4):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
        self.num_parts = num_parts
        self.processor = CLIPImageProcessor.from_pretrained(model_name)

    def forward(self, pil_image):
        # x: (B, 3, H, W)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        feat = self.model(**inputs).last_hidden_state  # (1, N, C)


                # Convert tokens ? feature map
        B, N, C = feat.shape
        H = W = int((N - 1) ** 0.5)
        fmap = feat[:, 1:, :].reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Split into parts
        part_h = H // self.num_parts
        parts = []
        for p in range(self.num_parts):
            h0 = p * part_h
            h1 = H if p == self.num_parts - 1 else (p + 1) * part_h
            pooled = fmap[:, :, h0:h1, :].mean(dim=[2, 3])
            parts.append(pooled)

        emb = torch.cat(parts, dim=1)
        emb = emb / emb.norm(dim=1, keepdim=True)
        return emb

def get_sam3_results(image, model, processor, text_prompt,device):
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

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                    
def crop_from_mask(image, mask, bbox):
    x0, y0, x1, y1 = map(int, bbox)

    # crop the region of interest
    crop = image[y0:y1, x0:x1].copy()
    #print(x0, y0, x1, y1)
    
    # crop the mask to the same region
    mask_crop = mask[y0:y1, x0:x1]

    # apply mask: zero out background
    crop[~mask_crop.astype(bool)] = 0

    return crop

def load_partsbasedencoding(crop1, crop2, parts_modelname, num_parts):
  clip_embed_obj = PartBasedEmbedding(parts_modelname, num_parts)
   
  ref_embed = clip_embed_obj.forward(crop1).detach().numpy()
  ref_embed = ref_embed.reshape(-1)

  img2_embed = clip_embed_obj.forward(crop2).detach().numpy()
  img2_embed = img2_embed.reshape(-1)
  
  partsbased_sim_score = cosine_similarity(ref_embed, img2_embed)
  return  partsbased_sim_score

def lightglue_geometric_similarity(mkpts0, mkpts1, scores=None,
                                   sigma=4.0, ransac_thresh=3.0):

    # Ensure correct shape
    mkpts0 = np.asarray(mkpts0).reshape(-1, 2).astype(np.float32)
    mkpts1 = np.asarray(mkpts1).reshape(-1, 2).astype(np.float32)

    # Ensure equal number of matches
    N = min(len(mkpts0), len(mkpts1))
    mkpts0 = mkpts0[:N]
    mkpts1 = mkpts1[:N]
    if scores is not None:
        scores = scores[:N]

    if N < 4:
        return 0.0
    
    # RANSAC homography
    H, inlier_mask = cv2.findHomography(
        mkpts0, mkpts1, cv2.RANSAC, ransacReprojThreshold=ransac_thresh
    )

    if H is None:
        return 0.0

    inlier_mask = inlier_mask.flatten().astype(bool)
    
    p_in = mkpts0[inlier_mask]
    q_in = mkpts1[inlier_mask]

    if len(p_in) == 0:
        return 0.0

    # Geometric error
    errors = np.linalg.norm(p_in - q_in, axis=1)
    geom_weights = np.exp(-(errors ** 2) / (sigma ** 2))

    # Combine with LightGlue confidence
    if scores is not None:
        weights = geom_weights * scores[inlier_mask]
    else:
        weights = geom_weights

    # Normalize by total matches
    score = weights.sum() / N
    return float(np.clip(score, 0.0, 1.0))

def load_lightglue(crop1, crop2):

  # =============================================================================
  # Load extractor and mat
  # =============================================================================

  # Read the first image and resize

  # SuperPoint+LightGlue
  extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
  matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

  img0 = torch.from_numpy(crop1).float() / 255.0
  img0 = img0.permute(2, 0, 1)[None].cuda()

  img1 = torch.from_numpy(crop2).float() / 255.0
  img1 = img1.permute(2, 0, 1)[None].cuda()

  with torch.no_grad():
      feats0 = extractor.extract(img0)
      feats1 = extractor.extract(img1)

      out = matcher({
          "image0": feats0,
          "image1": feats1,
      })

  feats0, feats1, out = [
      rbd(x) for x in [feats0, feats1, out]
  ]

  # Raw keypoints from SuperPoint
  kpts0 = feats0["keypoints"].detach().cpu().numpy()
  kpts1 = feats1["keypoints"].detach().cpu().numpy()

  # LightGlue match indices (correct fields)
  matches0 = out["matches0"].detach().cpu().numpy()            # shape (N0,)
  mscores0 = out["matching_scores0"].detach().cpu().numpy()    # shape (N0,)

  # Valid matches: matches0[i] = index into kpts1 or -1
  valid = matches0 > -1

  # Matched keypoints (aligned!)
  mkpts0 = kpts0[valid]
  mkpts1 = kpts1[matches0[valid]]
  scores = mscores0[valid]
      
  lg_sim_score = lightglue_geometric_similarity(mkpts0, mkpts1, scores)
  return lg_sim_score
 
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

def combined_similarity(cropA, cropB, clip_model_name, num_parts, y=3, a=0.8, b=0.15, c=0.05):
  part_sim_score = load_partsbasedencoding(cropA, cropB, clip_model_name, num_parts)
  lg_sim_score = load_lightglue(cropA, cropB)
  hu_sim_score = load_hu(cropA, cropB, y)
  
  total_simularity_score =  a*part_sim_score + b*lg_sim_score + c*hu_sim_score
  
  print(f"part based CLIP encoding score: {part_sim_score}")
  print(f"LightGlue similarity score: {lg_sim_score}")
  print(f"Hu similarity score: {hu_sim_score}")
  print(f"total similarity score: {total_simularity_score}")
  return total_simularity_score

def build_similarity_matrix(cropsA, cropsB, clip_model_name, num_parts, y=3, a=0.8, b=0.15, c=0.05):

    nA = len(cropsA)
    nB = len(cropsB)

    S = np.zeros((nA, nB), dtype=np.float32)

    for i in range(nA):
        for j in range(nB):
            S[i, j] = combined_similarity(
                cropsA[i], cropsB[j],clip_model_name, 
                num_parts, y=y, a=a, b=b, c=c)

    return S

def random_color():
    # Generate bright colors by sampling from 128–255 range
    return tuple(np.random.randint(128, 256, size=3).tolist())
                                   
def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    overlay = image.copy()
    color_layer = np.zeros_like(image)
    color_layer[:] = color

    overlay[mask > 0] = (
        alpha * color_layer[mask > 0] +
        (1 - alpha) * image[mask > 0]
    ).astype(np.uint8)

    return overlay

def display_figures(image1, image2, im1_title, im2_title, img_path):
    plt.figure(figsize=(12,6))

    # Left image
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title(im1_title)
    plt.axis("off")

    # Right image
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title(im2_title)
    plt.axis("off")

    plt.tight_layout()
    print(image_path)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
  # =============================================================================
  # If you want to run this script without errors, you will need a huggingface token 
  # that you add as an environment variable. Either name it HUGGINGFACE_HUB_TOKEN
  # or change the name in line 264 of the code below to match the name that you
  # chose for your environment variable. os will pull it from your environment.
  
  hf_token = os.environ["HUGGINGFACE_HUB_TOKEN"]
  login(hf_token)
 
  # =============================================================================
  # Load Configuration from YAML File
  # =============================================================================

  # Open and parse the YAML configuration file.
  with open("sceneREID_config.yaml", "r") as f:
      config = yaml.safe_load(f)

  # Assign configuration parameters to variables
  PROJECT_NAME = config["PROJECT_NAME"]
  EXPERIMENT_NAME = config["EXPERIMENT_NAME"]
  IMG0_PTH = Path(config["IMG0_PTH"])
  IMG1_PTH = Path(config["IMG1_PTH"])
  TEXT_PROMPT = config["TEXT_PROMPT"]
  SAM_MODEL_CFG = config["SAM_MODEL_ID"]
  RESIZE_HEIGHT = config["RESIZE_HEIGHT"]
  RESIZE_WIDTH = config["RESIZE_WIDTH"]
  VISUALIZE_FIG = config["VISUALIZE_FIG"]
  CLIP_MODEL_NAME = config["VIT_MODEL_NAME"]
  NUM_PARTS = config['NUM_PARTS']
  Y = config["Y"]
  A = config["A"]
  B = config["B"]
  C = config["C"]
  
  if VISUALIZE_FIG:
    # make savedir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    folder_name = os.path.join(PROJECT_NAME, EXPERIMENT_NAME, timestamp)
    
    os.makedirs(folder_name, exist_ok=True)

  
  if torch.cuda.is_available():
      DEVICE = torch.device('cuda')
  else:
      DEVICE = torch.device('cpu')

  img_ref_pil = Image.open(IMG0_PTH).convert("RGB")
  img_ref_np = np.array(img_ref_pil)   # H W 3 uint8

  img_pil = Image.open(IMG1_PTH).convert("RGB")
  img_np = np.array(img_pil)   # H W 3 uint8

  # Load the model
  sam_model = Sam3Model.from_pretrained(SAM_MODEL_CFG).to(DEVICE)
  sam_processor = Sam3Processor.from_pretrained(SAM_MODEL_CFG)

  # Segment using text prompt
  sam3_results_ref = get_sam3_results(img_ref_pil, sam_model, sam_processor, TEXT_PROMPT, DEVICE)
  sam3_results_img = get_sam3_results(img_pil, sam_model, sam_processor, TEXT_PROMPT, DEVICE)

  masks_ref  = sam3_results_ref["masks"].cpu().numpy()
  boxes_ref  = sam3_results_ref["boxes"].cpu().numpy()
  scores_ref = sam3_results_ref["scores"].cpu().numpy()
  
  masks_img  = sam3_results_img["masks"].cpu().numpy()
  boxes_img  = sam3_results_img["boxes"].cpu().numpy()
  scores_img = sam3_results_img["scores"].cpu().numpy()
  
  crop_ref_list = []
  crop_img_list = []
  
  for i in range(len(masks_ref)):
    crop_ref = crop_from_mask(img_ref_np, masks_ref[i], boxes_ref[i])
    crop_ref_list.append(crop_ref)
  for j in range(len(masks_img)):
    crop_img = crop_from_mask(img_np, masks_img[j], boxes_img[j])
    crop_img_list.append(crop_img)


  sim_mat = build_similarity_matrix(crop_ref_list, crop_img_list, CLIP_MODEL_NAME, 
            NUM_PARTS, y=Y, a=A, b=B, c=C)
  
  print(sim_mat)

  cost_matrix = -sim_mat

  row_ind, col_ind = linear_sum_assignment(cost_matrix)
  matches = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]
  print(f"Matches: {matches}")
  matched_scores = sim_mat[row_ind, col_ind]
  print(f"Matched scores : {matched_scores}")
  
  if VISUALIZE_FIG:
    # original images
    
    first_title = "Unaltered Reference Image"
    second_title = "Unaltered Second Image"
    image_name = f"{EXPERIMENT_NAME}_{timestamp}_original_images.png"
    image_path = os.path.join(folder_name, image_name)

    display_figures(img_ref_np, img_np, first_title, second_title, image_path)
    
    # original images with masks
    
    ref_combined = img_ref_np.copy()
    img_combined = img_np.copy()

    for i, mask in enumerate(masks_ref):
        color = random_color()
        ref_combined = overlay_mask(ref_combined, mask, color, alpha=0.8)

    for i, mask in enumerate(masks_img):
        color = random_color()
        img_combined = overlay_mask(img_combined, mask, color, alpha=0.8)
    
    first_title = "Reference Image with masks"
    second_title = "Second Image with masks"
    image_name = f"{EXPERIMENT_NAME}_{timestamp}_imagesANDmasks.png"
    image_path = os.path.join(folder_name, image_name)

    display_figures(ref_combined, img_combined, first_title, second_title, image_path)
  
    # matches with masks
    
    ref_matches_combined = img_ref_np.copy()
    img_matches_combined = img_np.copy()

    for amatch in matches:
      color = random_color()
      ref_matches_combined = overlay_mask(ref_matches_combined, masks_ref[amatch[0]], color, alpha=0.8)
      img_matches_combined = overlay_mask(img_matches_combined, masks_img[amatch[1]], color, alpha=0.8)
      
    first_title = "Reference Image with matches"
    second_title = "Second Image with matches"
    image_name = f"{EXPERIMENT_NAME}_{timestamp}_imagesWITHmatches.png"
    image_path = os.path.join(folder_name, image_name)

    display_figures(ref_matches_combined, img_matches_combined, first_title, second_title, image_path)
