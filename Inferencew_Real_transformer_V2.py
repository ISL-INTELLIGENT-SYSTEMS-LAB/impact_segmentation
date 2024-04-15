#!/usr/bin/env python
'''
#################################################################################################
#
#
#   Fayetteville State University Intelligence Systems Laboratory (FSU-ISL)
#
#   Mentors:
#           Dr. Sambit Bhattacharya
#
#   File Name:
#          IMPACT_transformer.py
#
#   Programmers:
#           Catherine Spooner
#           Carley Brinkley
#           
#
#
#  Revision     Date                        Release Comment
#  --------  ----------  ------------------------------------------------------
#    1.0     10/30/2023  Initial Release
#
#  File Description
#  ----------------
#  

#
#
#  *Classes/Functions organized by order of appearance
#
#  OUTSIDE FILES REQUIRED
#  ----------------------
#   None
#
#  CLASSES
#  -------
#   TransformerDataSet
#
#  FUNCTIONS
#  ---------

#
'''
#################################################################################################
#Import Statements
#################################################################################################
import os
import sys

# extremely important that you change these variables for your particular setup.
VIR_ENV_DIR = 'v_env'
VIR_ENV_NAME = 'maskformer'
PYTHON_TYPE = 'python3.7'
user = os.getlogin()
# I put my virtual environment in /home/cspooner/v_env/maskformer

virenv_libs = f'/home/{user}/{VIR_ENV_DIR}/{VIR_ENV_NAME}/lib/{PYTHON_TYPE}/site-packages'


if virenv_libs not in sys.path:
    sys.path.insert(0,virenv_libs)

#####################################################################################

import random
import torch
import evaluate

import filetype
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
import pandas as pd

from datetime import datetime
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from torch import nn
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    MaskFormerConfig,
    MaskFormerImageProcessor,
    MaskFormerModel,
    MaskFormerForInstanceSegmentation,
)

#from huggingface_hub import notebook_login
from IMPACT_transformer_V2 import *

if __name__ == "__main__":
  print(virenv_libs)
  print("Completed import libraries.")

def main():

    NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
    ID2LABEL = {0:'soil', 1:'bedrock', 2:'sand', 3:'bigrock', 255:'unknown'}
    LABEL2ID = {'soil':0, 'bedrock':1, 'sand':2, 'bigrock':3, 'unknown':255}
    RANDOM_SEED = 2024
    
    label2color = {255:(0,0,0), 0:(107,113,115), 1:(92,189,224), 2:(20,201,150), 3:(209,65,20)}

    PROJECTDIR = 'IMPACT'
    subproject = 'ai4mars'
    DATADIR = f'/home/{user}/{PROJECTDIR}/ai4mars-dataset-merged-0.1/re_encoded_files'
    SAVEROOT = f'/home/{user}/{PROJECTDIR}/{subproject}'
    
    IMGDIR = 'images_train'
    IFRAG = 'realimage'
    SEGDIR = 'labels_train'
    SFRAG = 'segment'
    IMFIX = '.JPG'
    LABFIX = '.png'
    IMAGE_SIZE_X = 512
    IMAGE_SIZE_Y = 512
    THRESHOLD = 0.5

    # Define the name of the mode
    MODEL_NAME = "facebook/maskformer-swin-base-ade"
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 4
    MODEL_LOCATION = "/home/cspooner/IMPACT/ai4mars/weights_20240410_154853/"
    SAVE_RESULTS_LOCATION = MODEL_LOCATION.replace("weights", "results")
    NUM_INFERENCES = 10

    ###
    # NEED TO GET THE MEAN AND STD OF DATASET
    ###

    
    #create dataset
    tf_dataset = TransformerDataSet(DATADIR, IMGDIR, SEGDIR, IFRAG, SFRAG, IMFIX, LABFIX)

    training_val_dataset = tf_dataset.train_val_dataset(random_state=RANDOM_SEED)
    train = training_val_dataset["train"]
    validation = training_val_dataset["validation"]
    test = training_val_dataset["test"]

    print(len(train))
    print(len(validation))
    print(len(test))


    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Grab the trained model and processor from local files
    model = MaskFormerForInstanceSegmentation.from_pretrained(MODEL_LOCATION, local_files_only=True).to(device)
    processor = MaskFormerImageProcessor.from_pretrained(MODEL_LOCATION, local_files_only=True)

    #pick NUM_INFERENCES random indices to run inference on
    indices = random.sample(range(len(test)), NUM_INFERENCES)
    print(indices)
    
    if len(indices) > 0:
      for index in indices:
        # Use random test image
        image = test[index]["image"].convert("RGB")
        filename = test[index]['filename']
        result_filename = filename.replace(IMFIX, "")
        print(f'image with index {index} has filename of {filename}')
        target_size = image.size[::-1]
        # Preprocess image
        inputs = processor(images=image, return_tensors="pt").to(device)
        # Inference
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
    
        # Let's print the items returned by our model and their shapes
        print("[INFO] Displaying shape of the outputs...")
        for key, value in outputs.items():
            print(f"  {key}: {value.shape}")
    
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
    
        instance_seg_mask_disp = visualize_instance_seg_mask(instance_seg_mask, label2color)
        groundTruthAnn = np.array(test[index]["annotation"])[:,:,0]
        groundtruth_seg_mask_disp = visualize_instance_seg_mask(groundTruthAnn, label2color)
        
        plt.figure(figsize=(10, 10))
        plt.style.use('_mpl-gallery-nogrid')
        for plot_index in range(3):
            if plot_index == 0:
                plot_image = image
                title = "Original Image"
            elif plot_index == 1:
                plot_image = groundtruth_seg_mask_disp
                title = 'Ground Truth Annotation'
            else:
                plot_image = instance_seg_mask_disp
                title = "Predicted Segmentation"
    
            plt.subplot(1, 3, plot_index+1, label=title)
            plt.title(title)
        plt.imshow(plot_image)
        plt.legend()
        plt.axis("off")   
        plt.savefig(os.path.join(SAVE_RESULTS_LOCATION, f"{NOW}_trainingresult_{result_filename}.png"), dpi = 300, bbox_inches='tight')
        
    '''
    # Load Mean IoU metric
    metrics = evaluate.load("mean_iou")
    # Set model in evaluation mode
    model.eval()
    # Test set doesn't have annotations so we will use the validation set
    ground_truths, preds = [], []
    for idx in tqdm(range(len(validation))):
        image = validation[idx]["image"].convert("RGB")
        target_size = image.size[::-1]
        # Get ground truth semantic segmentation map
        annotation = np.array(validation[idx]["annotation"])[:,:,0]
        print(np.unique(annotation))
        # Replace null class (0) with the ignore_index (255) and reduce labels
        annotation -= 1
        annotation[annotation==-1] = 255
        ground_truths.append(annotation)
        # Preprocess image
        inputs = processor(images=image, return_tensors="pt").to(device)
        # Inference
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process results to retrieve semantic segmentation maps
        result = processor.post_process_semantic_segmentation(outputs, target_sizes=[target_size])[0]
        semantic_seg_mask = result.cpu().detach().numpy()
        preds.append(semantic_seg_mask)
    print("completed stuff")
    results = metrics.compute(
        predictions=preds,
        references=ground_truths,
        num_labels=100,
        ignore_index=255
    )
    print(f"Mean IoU: {results['mean_iou']} | Mean Accuracy: {results['mean_accuracy']} | Overall Accuracy: {results['overall_accuracy']}")
    '''

if __name__ == "__main__":
    main()