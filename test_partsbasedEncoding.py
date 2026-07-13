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
    print(x0, y0, x1, y1)
    
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

  #print(sam3_results)
  masks_ref  = sam3_results_ref["masks"].cpu().numpy()
  masks_img  = sam3_results_img["masks"].cpu().numpy()

  boxes_ref  = sam3_results_ref["boxes"].cpu().numpy()
  boxes_img  = sam3_results_img["boxes"].cpu().numpy()

  scores_ref = sam3_results_ref["scores"].cpu().numpy()
  scores_img = sam3_results_img["scores"].cpu().numpy()

  crop_ref = crop_from_mask(img_ref_np, masks_ref[0], boxes_ref[0])
  crop_img = crop_from_mask(img_np, masks_img[0], boxes_img[0])

  part_simscore = load_partsbasedencoding(crop_ref, crop_img, CLIP_MODEL_NAME, NUM_PARTS)
  print(part_simscore)