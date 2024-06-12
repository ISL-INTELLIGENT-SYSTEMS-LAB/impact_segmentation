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
from sklearn.metrics import jaccard_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

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
#import argparse

#from huggingface_hub import notebook_login
from IMPACT_transformer_V2 import *

if __name__ == "__main__":
  print(virenv_libs)
  print("Completed import libraries.")

def main():

    NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
    #ID2LABEL = {0:'soil', 1:'bedrock', 2:'sand', 3:'bigrock', 255:'unknown'}
    #LABEL2ID = {'soil':0, 'bedrock':1, 'sand':2, 'bigrock':3, 'unknown':255}
    RANDOM_SEED = 2024
    
    #label2color = {255:(0,0,0), 0:(107,113,115), 1:(92,189,224), 2:(20,201,150), 3:(209,65,20)}
    #lab_names = ['background', 'soil', 'bedrock', 'sand', 'big_rock']
    #print(label2color.keys())
    PROJECTDIR = 'IMPACT'
    subproject = 'ai4mars'
    DATADIR = f'/home/{user}/{PROJECTDIR}/ai4mars-dataset-merged-0.1/re_encoded_files_V2'
    #DATADIR = f'/home/{user}/{PROJECTDIR}/Data_Collection/ai4mars-dataset-merged-0.1/tiny_dataset'
    SAVEROOT = f'/home/{user}/{PROJECTDIR}/{subproject}'
    
    TRAIN_IMGDIR = 'images_train'
    TEST_IMGDIR = 'images_test'
    IFRAG = 'realimage'
    TRAIN_SEGDIR = 'labels_train'
    TEST_SEGDIR = 'labels_test'
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
    MODEL_LOCATION = f"/home/{user}/{PROJECTDIR}/{subproject}/weights_20240425_140220/"
    SAVE_RESULTS_LOCATION = MODEL_LOCATION.replace("weights", "results")
    #NUM_INFERENCES_SHOW = 20
    NUM_INFERENCES_SHOW = 0
    #dataloader_path = "20240507_141435_dataloader.pth"
    dataloader_path = None
    #use_datadirpath_for_data = True
    

    ###
    # NEED TO GET THE MEAN AND STD OF DATASET
    ###

    #create dataset
    
    
    print("Creating test dataset...")
    test = TransformerDataSet(DATADIR, TEST_IMGDIR, TEST_SEGDIR, IFRAG, SFRAG, IMFIX, LABFIX)


    print(len(test))


    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Grab the trained model and processor from local files
    model = MaskFormerForInstanceSegmentation.from_pretrained(MODEL_LOCATION, local_files_only=True).to(device)
    processor = MaskFormerImageProcessor.from_pretrained(MODEL_LOCATION, local_files_only=True)

    #pick NUM_INFERENCES random indices to run inference on
    indices = random.sample(range(len(test)), NUM_INFERENCES_SHOW)
    print(indices)
    
    # Load Mean IoU metric
    metrics = evaluate.load("mean_iou")
    #confusion_metric = evaluate.load("confusion_matrix")
    
    ground_truths = []
    preds = []
    total_pixels = 0
    conf_mat = np.zeros((5,5))
    
    for idx in tqdm(range(len(test))):
        testObj = test[idx]
        filename = testObj['filename']
        result_filename = filename.replace(IMFIX, "")
        
        semantic_seg_mask = get_inference(testObj, processor, device, model, IMFIX)
        conf_semsegmask = semantic_seg_mask.copy()
        conf_semsegmask[conf_semsegmask==-1] = 4
        semantic_seg_mask[semantic_seg_mask==-1] = 255

        semantic_seg_mask = semantic_seg_mask.astype("int32")

        annotation = np.array(testObj["annotation"])[:,:,0]
        # Replace null class (0) with the ignore_index (255) and reduce labels
        annotation -= 1
        conf_ann = annotation.copy()
        conf_ann[conf_ann==255] = 4
        #annotation[annotation==-1] = 255
        annotation = annotation.astype("int32")
        #print(type(annotation))
         
        ground_truths.append(annotation)
        preds.append(semantic_seg_mask)
        #print(f'ground truth - {np.unique(annotation)}')
        #print(f'ground truth - {np.unique(conf_ann)}')
        #print(f'prediction - {np.unique(conf_semsegmask)}')
        
        cm = confusion_matrix(conf_ann.flatten(), conf_semsegmask.flatten(), labels=list(range(5)))
        conf_mat = conf_mat + cm
        total_pixels += len(conf_ann.flatten())
        
        
        #print(cm)
        
        if idx in indices:
          seg_mask_disp = visualize_instance_seg_mask(semantic_seg_mask, label2color, "semseg")
          groundtruth_seg_mask_disp = visualize_instance_seg_mask(annotation, label2color, "groundtruth")
          plt.figure(figsize=(10, 10))
          plt.style.use('_mpl-gallery-nogrid')
          for plot_index in range(3):
              if plot_index == 0:
                  plot_image = testObj["image"].convert("RGB")
                  title = "Original Image"
              elif plot_index == 1:
                  plot_image = groundtruth_seg_mask_disp
                  title = 'Ground Truth Annotation'
              else:
                  plot_image = seg_mask_disp
                  title = "Predicted Segmentation"
      
              plt.subplot(1, 3, plot_index+1)
              plt.title(title)
              plt.imshow(plot_image)
          #plt.legend()
          plt.axis("off")   
          plt.savefig(os.path.join(SAVE_RESULTS_LOCATION, f"trainingresult_{result_filename}.jpg"), dpi = 300, bbox_inches='tight')
  
    print("completed stuff")

    results = metrics.compute(predictions=preds, references=ground_truths, num_labels=5, ignore_index=255)
    print(f"Mean IoU: {results['mean_iou']} | Mean Accuracy: {results['mean_accuracy']} | Overall Accuracy: {results['overall_accuracy']}")
    print(results)
    
    #print(conf_mat/total_pixels)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=(conf_mat/total_pixels))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Semantic Segmentation")
    plt.savefig(os.path.join(SAVE_RESULTS_LOCATION, "confusion_matrix_results.jpg"), dpi = 300, bbox_inches='tight')
    #plt.show()
    
if __name__ == "__main__":                    
                    
    main()
