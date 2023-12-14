#!/usr/bin/env python
# coding: utf-8


import sys
print(sys.path)
sim_packages = "/home/cbrinkley/vir_env/simulation/lib/python3.7/site-packages"
if sim_packages not in sys.path:
    sys.path.insert(0,sim_packages)
print(sys.path)


# Import the necessary packages
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch import nn
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
import albumentations as A
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    MaskFormerConfig,
    MaskFormerImageProcessor,
    MaskFormerModel,
    MaskFormerForInstanceSegmentation,
)
import evaluate
#from huggingface_hub import notebook_login
from datetime import datetime
import os
import filetype
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import cv2
print("Completed import libraries.")

def visualize_instance_seg_mask(mask):
    # Initialize image with zeros with the image resolution
    # of the segmentation mask and 3 channels
    image = np.zeros((mask.shape[0], mask.shape[1], 3))
    # Create labels
    labels = np.unique(mask)
    label2color = {-1:(0,0,0), 0:(89, 89, 89), 1:(255, 255, 0), 2:(0,255,0), 3:(0, 139, 231)}
    for height in range(image.shape[0]):
        for width in range(image.shape[1]):
            image[height, width, :] = label2color[mask[height, width]]
    image = image / 255
    return image

savelocation = "/home/cbrinkley/sims_projects/"
label2color = {-1:(0,0,0), 0:(89, 89, 89), 1:(255, 255, 0), 2:(0,255,0), 3:(0, 139, 231)}
tstimg_loc = os.path.join(savelocation, "data_collection", "mars_perseverance")
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Grab the trained model and processor from the hub
modelname = "pytorch_model.bin"
model = MaskFormerForInstanceSegmentation.from_pretrained(savelocation,local_files_only = True).to(device)
processor = MaskFormerImageProcessor.from_pretrained(savelocation,local_files_only = True)


tstimg_lst = os.listdir(tstimg_loc)
tstimg_lst = [x for x in tstimg_lst if x.endswith('.png')]
for i in tstimg_lst:
    #img = Image.open(file_path)
    image = Image.open(os.path.join(tstimg_loc, i)).convert("RGB")
    target_size = image.size[::-1]
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt").to(device)
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)


    # Let's print the items returned by our model and their shapes
    #print("Outputs...")
    #for key, value in outputs.items():
        #print(f"  {key}: {value.shape}")


    # Post-process results to retrieve instance segmentation maps
    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        target_sizes=[target_size]
    )[0] # we pass a single output therefore we take the first result (single)
    instance_seg_mask = result["segmentation"].cpu().detach().numpy()
    print(f"Final mask shape: {instance_seg_mask.shape}")
    print("Segments Information...")
    for info in result["segments_info"]:
        print(f"  {info}")


    instance_seg_mask_disp = visualize_instance_seg_mask(instance_seg_mask)
    plt.figure(figsize=(10, 4))
    for plot_index in range(2):
        if plot_index == 0:
            plot_image = image
            title = "Original"
        else:
            plot_image = instance_seg_mask_disp
            values = np.unique(plot_image)
            title = "Segmentation"

        plt.subplot(1, 2, plot_index+1)
        im = plt.imshow(plot_image)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(savelocation, f"pic_after_training_{i}.png"), dpi = 300)
    #plt.show()


