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
import cv2

class PartBasedEmbedding(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", num_parts=4):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)
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

def visualize_lightglue_matches(
    crop0,
    crop1,
    kpts0,
    kpts1,
    matches,
    scores=None,
    max_matches=100,
    title="LightGlue matches",
):
    """
    crop0, crop1: RGB numpy arrays, H x W x 3
    kpts0, kpts1: keypoints, shape [N, 2]
    matches: matched index pairs, shape [M, 2]
             where each row is [idx_in_kpts0, idx_in_kpts1]
    scores: optional match confidence scores, shape [M]
    """

    img0 = crop0.copy()
    img1 = crop1.copy()

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    canvas_h = max(h0, h1)
    canvas_w = w0 + w1

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0 + w1] = img1

    if scores is not None:
        order = np.argsort(scores)[::-1]
        matches = matches[order]
        scores = scores[order]

    matches = matches[:max_matches]

    plt.figure(figsize=(14, 8))
    plt.imshow(canvas)
    plt.axis("off")
    plt.title(title)

    for i, (idx0, idx1) in enumerate(matches):
        x0, y0 = kpts0[idx0]
        x1, y1 = kpts1[idx1]
        x1 = x1 + w0

        plt.plot([x0, x1], [y0, y1], linewidth=0.8)
        plt.scatter([x0, x1], [y0, y1], s=8)

    plt.show()

def visualize_lightglue_inliers(
    img0, img1,
    mkpts0, mkpts1,
    scores,
    inlier_mask,
    cmap='viridis',
    point_size=8,
    line_width=2,
):
    """
    Visualize only RANSAC inliers, color-coded by LightGlue confidence.
    """

    # Convert images to RGB if needed
    if img0.ndim == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)

    # Concatenate images horizontally
    H0, W0 = img0.shape[:2]
    H1, W1 = img1.shape[:2]
    canvas = np.zeros((max(H0, H1), W0 + W1, 3), dtype=np.uint8)
    canvas[:H0, :W0] = img0
    canvas[:H1, W0:W0 + W1] = img1

    # Extract inliers
    p_in = mkpts0[inlier_mask]
    q_in = mkpts1[inlier_mask]
    s_in = scores[inlier_mask]

    # Normalize scores for colormap
    s_norm = (s_in - s_in.min()) / (s_in.max() - s_in.min() + 1e-8)
    colors = plt.cm.get_cmap(cmap)(s_norm)[:, :3]  # RGB

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(canvas)
    ax.axis('off')

    # Draw matches
    for (x0, y0), (x1, y1), color in zip(p_in, q_in, colors):
        # Shift x1 by width of left image
        x1_shifted = x1 + W0

        ax.plot(
            [x0, x1_shifted],
            [y0, y1],
            color=color,
            linewidth=line_width,
            alpha=0.8
        )

        ax.scatter(
            [x0, x1_shifted],
            [y0, y1],
            color=color,
            s=point_size
        )

    plt.tight_layout()
    return fig

def lightglue_geometric_similarity(mkpts0, mkpts1, scores=None,
                                   sigma=4.0, ransac_thresh=3.0, visFig=False, matching_name="lightglue"):


    now = datetime.now()

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
    
    '''if visFig:
      fig = visualize_lightglue_inliers(
      crop_ref,
      crop_img,
      mkpts0,
      mkpts1,
      scores,
      inlier_mask,
      )
      
      fig.savefig(f"{matching_name}_inliers.png", dpi=300, bbox_inches="tight")'''

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

'''def size_similarity(maskA, maskB):
    areaA = float(np.sum(maskA))
    areaB = float(np.sum(maskB))

    # Avoid division by zero
    if areaA == 0 or areaB == 0:
        return 0.0

    diff = abs(areaA - areaB)
    max_area = max(areaA, areaB)

    sim = 1.0 - (diff / max_area)
    return float(np.clip(sim, 0.0, 1.0))'''

'''def color_hist_similarity(cropA, cropB,
                          hist_size=(32, 32),
                          ranges=[0, 180, 0, 256]):
    """
    Compute color histogram similarity between two masked crops.
    Uses HSV histograms and correlation metric.
    
    Parameters:
        cropA, cropB : np.ndarray (H,W,3) RGB or BGR images
        maskA, maskB : np.ndarray (H,W) boolean or 0/1 masks
        hist_size    : bins for H and S channels
        ranges       : histogram ranges for H and S
        
    Returns:
        similarity score in [0, 1]
    """

    # Convert to HSV
    hsvA = cv2.cvtColor(cropA, cv2.COLOR_BGR2HSV)
    hsvB = cv2.cvtColor(cropB, cv2.COLOR_BGR2HSV)

    # If masks are None, use full image
    maskA = np.ones(hsvA.shape[:2], dtype=np.uint8)
    maskB = np.ones(hsvB.shape[:2], dtype=np.uint8)
    
    # Compute histograms for H and S channels
    histA = cv2.calcHist([hsvA], [0, 1], maskA, hist_size, ranges)
    histB = cv2.calcHist([hsvB], [0, 1], maskB, hist_size, ranges)

    # Normalize histograms
    cv2.normalize(histA, histA)
    cv2.normalize(histB, histB)

    # Compare using correlation (range [-1, 1])
    sim = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)

    # Convert correlation to [0, 1]
    sim = (sim + 1) / 2.0

    return float(sim)'''

    
def combined_similarity(cropA, cropB, clip_model_name, num_parts, a=0.6, b=0.15, c=0.05, d=0.1, e=0.1):
  part_sim_score = load_partsbasedencoding(cropA, cropB, clip_model_name, num_parts)
  
  lg_sim_score = load_lightglue(cropA, cropB)

  hu_sim_score = load_hu(cropA, cropB, Y)
  
  #size_sim_score = size_similarity(cropA, cropB)
  
  #color_sim_score = color_hist_similarity(cropA, cropB)
  
  total_simularity_score =  a*part_sim_score + b*lg_sim_score + c*hu_sim_score
  '''total_simularity_score =  a*part_sim_score + b*lg_sim_score + c*hu_sim_score \
                            + d*size_sim_score + e*color_sim_score'''
  
  return total_simularity_score

def build_similarity_matrix(cropsA, cropsB, clip_model_name, num_parts, a=0.6, b=0.15, c=0.05, d=0.1, e=0.1):

    nA = len(cropsA)
    nB = len(cropsB)

    S = np.zeros((nA, nB), dtype=np.float32)

    for i in range(nA):
        for j in range(nB):
            S[i, j] = combined_similarity(
                cropsA[i], cropsB[j],clip_model_name, 
                num_parts, a=a, b=b, c=c, d=d, e=e
            )

    return S
    
if __name__ == "__main__":

  hf_token = os.environ["HUGGINGFACE_HUB_TOKEN"]
  login(hf_token)
 
  # =============================================================================
  # Load Configuration from YAML File
  # =============================================================================

  # Open and parse the YAML configuration file.
  with open("sceneREID_config.yaml", "r") as f:
      config = yaml.safe_load(f)

  # Assign configuration parameters to variables
  EXPERIMENT_NAME = config["EXPERIMENT_NAME"]
  IMG0_PTH = Path(config["IMG0_PTH"])
  IMG1_PTH = Path(config["IMG1_PTH"])
  TEXT_PROMPT = config["TEXT_PROMPT"]
  SAM_MODEL_CFG = config["SAM_MODEL_ID"]
  MATCHING_NAME = config["MATCHING_NAME"]
  RESIZE_HEIGHT = config["RESIZE_HEIGHT"]
  RESIZE_WIDTH = config["RESIZE_WIDTH"]
  CREATE_SAVEDIR = config["CREATE_SAVEDIR"]
  VISUALIZE_FIG = config["VISUALIZE_FIG"]
  CLIP_MODEL_NAME = config["VIT_MODEL_NAME"]
  NUM_PARTS = config['NUM_PARTS']
  Y = config["Y"]
  A = config["A"]
  B = config["B"]
  C = config["C"]
  D = config["D"]
  E = config["E"]     
  if torch.cuda.is_available():
      DEVICE = torch.device('cuda')
  else:
      DEVICE = torch.device('cpu')

  img_ref_pil = Image.open(IMG0_PTH).convert("RGB")
  img_ref_np = np.array(img_ref_pil)   # H W 3 uint8

  img_pil = Image.open(IMG1_PTH).convert("RGB")
  img_np = np.array(img_pil)   # H W 3 uint8

    # Load the model
  model = Sam3Model.from_pretrained(SAM_MODEL_CFG).to(DEVICE)
  processor = Sam3Processor.from_pretrained(SAM_MODEL_CFG)

  # Segment using text prompt
  sam3_results_ref = get_sam3_results(img_ref_pil,TEXT_PROMPT,DEVICE)
  sam3_results_img = get_sam3_results(img_pil,TEXT_PROMPT,DEVICE)

 
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

  '''# Create a figure with 1 row and 2 columns
  # figsize=(width, height) sets the proportions of the overall window
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

  ax1.imshow(crop_ref)
  ax1.set_title('ref crop')

  ax2.imshow(crop_img)
  ax2.set_title('img crop')

  # 5. Clean up layout and display
  plt.tight_layout() # Prevents overlapping labels
  plt.show()'''

sim_mat = build_similarity_matrix(crop_ref_list, crop_img_list, CLIP_MODEL_NAME, NUM_PARTS, A, B, C, D, E)
print(sim_mat)