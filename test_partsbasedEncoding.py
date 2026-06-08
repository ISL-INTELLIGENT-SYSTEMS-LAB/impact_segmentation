import os
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPVisionModel, CLIPImageProcessor, Sam3Processor, Sam3Model
from PIL import Image
from huggingface_hub import login
import matplotlib.pyplot as plt

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


def get_sam3_results(image, text_prompt, model_name, device):
  model = Sam3Model.from_pretrained(model_name).to(device)
  processor = Sam3Processor.from_pretrained(model_name)

  # Load an image
  
  # Segment using text prompt
  
  inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device) 
  # Prompt the model with text
  with torch.no_grad():
      outputs = model(**inputs)
  
  # Post-process results
  results = processor.post_process_instance_segmentation(
      outputs,
      threshold=0.5,
      mask_threshold=0.5,
      target_sizes=inputs.get("original_sizes").tolist())[0]
  masks  = results["masks"].cpu().numpy().tolist()
  boxes  = results["boxes"].cpu().numpy().tolist()
  scores = results["scores"].cpu().numpy().tolist()
    
  return masks, boxes, scores

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


if __name__ == "__main__":

  hf_token = os.environ["HUGGINGFACE_HUB_TOKEN"]
  login(hf_token)
  
  #image_reference = r"E:\IMPACT\data_collection\scene_integration\easter_files\placement_data_20250603_130127\20250603_130127_Camera7_-1_1_270.png"
  image_reference = r"E:\IMPACT\data_collection\scene_integration\easter_files\placement_data_20250603_130127\20250603_130127_Camera5_-1_1_180.png"
  image2 = r"E:\IMPACT\data_collection\scene_integration\easter_files\placement_data_20250603_130127\20250603_130127_Camera23_-3_-1_90.png"
       
  TEXT_PROMPT = "orange egg" 
  SAM_MODEL_NAME = "facebook/sam3"
  CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
  NUM_PARTS = 6
  threshold_same = 0.8
  threshold_class = 0.5
  threshold_not = 0.3

  clip_embed_obj = PartBasedEmbedding(CLIP_MODEL_NAME, NUM_PARTS)
  
  device = "cuda" if torch.cuda.is_available() else "cpu"
  
  ref_pil = Image.open(image_reference).convert("RGB")
  ref_masks, ref_bbox, ref_scores = get_sam3_results(image_reference, TEXT_PROMPT, SAM_MODEL_NAME, device)
  ref_bbox = [int(x) for x in ref_bbox[0]]
  ref_crop = crop_from_mask(np.array(ref_pil), np.array(ref_masks[0]), ref_bbox)
  
  ref_embed = clip_embed_obj.forward(ref_crop).detach().numpy()
  ref_embed = ref_embed.reshape(-1)


  
  
  img2_pil = Image.open(image2).convert("RGB")
  img2_masks, img2_bbox, img2_scores = get_sam3_results(image2, TEXT_PROMPT, SAM_MODEL_NAME, device)
  
  img2_bbox = [int(x) for x in img2_bbox[0]]
  img2_crop = crop_from_mask(np.array(img2_pil), np.array(img2_masks[0]), img2_bbox)
  #plt.imshow(img2_crop)
  #plt.imshow(ref_crop)
  #plt.show()
  
  img2_embed = clip_embed_obj.forward(img2_crop).detach().numpy()
  img2_embed = img2_embed.reshape(-1)
  
  cos_sim = cosine_similarity(ref_embed, img2_embed)
  print(cos_sim)
  