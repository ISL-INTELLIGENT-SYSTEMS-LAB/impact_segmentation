import os
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import Sam3Processor, Sam3Model, CLIPProcessor, CLIPModel
from huggingface_hub import login

hf_token = os.environ["HUGGINGFACE_HUB_TOKEN"]
login(hf_token)


class ClipEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_image(self, crop_np):
        #print("\n=== CLIP DEBUG START ===")

        inputs = self.processor(
            images=crop_np,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        # Case 1: direct tensor
        if isinstance(outputs, torch.Tensor):
          img_emb = outputs

        # Case 2: BaseModelOutputWithPooling
        else:
          # Prefer pooled embedding if available
          if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            img_emb = outputs.pooler_output
          # Otherwise use CLS token from last_hidden_state
          elif hasattr(outputs, "last_hidden_state"):
            img_emb = outputs.last_hidden_state[:, 0, :]
          else:
            raise RuntimeError("Could not extract image embedding from CLIP output")
            
        # 5. Normalize embedding (recommended for similarity search)
        img_emb /= img_emb.norm(dim=-1, keepdim=True)
        
        #print(img_emb.shape) # e.g., torch.Size([1, 768])
        return img_emb[0].cpu().numpy()

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
    
    # crop the mask to the same region
    mask_crop = mask[y0:y1, x0:x1]

    # apply mask: zero out background
    crop[~mask_crop.astype(bool)] = 0

    return crop

   
# =============================================================================
# Load the Input Image Pair using pillow, and resize to same size used in LoFTR
# =============================================================================
REF_IMG_PTH = "/home/cspooner/IMPACT1/data_collection/scene_integration/placement_data_20250602_141151/20250602_141151_Camera_-5_5_135.png"
#20250602_141151_Camera_-5_5_135.png
IMG_DIR_PTH = "/home/cspooner/IMPACT1/data_collection/scene_integration/placement_data_20250602_141151"
TEXT_PROMPT = "orange egg" 
SAM_MODEL_NAME = "facebook/sam3"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# select the device for computation
device = "cuda" if torch.cuda.is_available() else "cpu"

'''
img0_pil = Image.open(IMG0_PTH).convert("RGB")
img1_pil = Image.open(IMG1_PTH).convert("RGB")

clip_model = ClipEmbedder(model_name=CLIP_MODEL_NAME, device=device)


# =============================================================================
# SAM3 Mask Generation
# =============================================================================


img0_masks, img0_bbox, img0_scores = get_sam3_results(img0_pil, TEXT_PROMPT, SAM_MODEL_NAME, device)
img1_masks, img1_bbox, img1_scores = get_sam3_results(img1_pil, TEXT_PROMPT, SAM_MODEL_NAME, device)
print(len(img0_masks))
print(len(img1_masks))
#print(sam3_results)


for i in range(len(img0_masks)):
    highest_similarity = [-1]
    mask0  = img0_masks[i]
    bbox0   = img0_bbox[i]
    score0 = img0_scores[i]
    label = TEXT_PROMPT#[i]   # SAM3 assigns the prompt index

    crop0 = crop_from_mask(np.array(img0_pil), mask0, bbox0)
    sem_emb0 = clip_model.embed_image(crop0)
    
    for j in range(len(img1_masks)):
      
      mask1 = img1_masks[i]
      bbox1 = img1_bbox[i]
      score1 = img1_scores[i]

      crop1 = crop_from_mask(np.array(img1_pil), mask1, bbox1)
      sem_emb1 = clip_model.embed_image(crop1)
      
      sim = cosine_similarity(sem_emb0, sem_emb1)

      if sim > highest_similarity[i]:
        highest_similarity[i] = sim
    
    
    for hsim in highest_similarity:
      if hsim >= 0.5:
        print(f"the highest similarity score is : {hsim}. The object is likely to be the same type in both images.")
      elif hsim < 0.5 and hsim >= 0.3:
        print(f"the highest similarity score is : {hsim}. This might be the same object. More tests need to be made")
      else:
        print(f"the highest similarity score is : {hsim}. This is probably not the same object")


'''
img_list = os.listdir(IMG_DIR_PTH)
img_list = [x for x in img_list if x.endswith("png")]

for img_name in img_list:
  img_path = os.path.join(IMG_DIR_PTH, img_name)
  img_pil = Image.open(img_path).convert("RGB")
  img0_masks, img0_bbox, img0_scores = get_sam3_results(img_pil, TEXT_PROMPT, SAM_MODEL_NAME, device)
  print(len(img0_masks), img_name )

      
      
