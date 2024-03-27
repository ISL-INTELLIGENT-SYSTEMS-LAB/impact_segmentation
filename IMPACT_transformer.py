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
#    2.0     03/27/2024  Bug fixes 
#
#  File Description
#  ----------------
#  
#  This program will train a model to make class predictions for each pixel in an image(semantic segmentation masks)
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
#   ImageSegmentationDataset
#
#  FUNCTIONS
#  ---------
#   create_image_fromdataset
#   collate_fn
#   visualize_instance_seg_mask
#   create_loss_plots
#   main
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
print(virenv_libs)

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
print("Completed import libraries.")


class TransformerDataSet(Dataset):
    def __init__(self, root_dir, img_dir=None, seg_dir=None, imgFrag=None, segFrag=None, imgfix=".png", segfix=".png"):
        images = []
        annotations = []
        file_name = []
        if img_dir is None:
            img_dir = 'realimage'
        if seg_dir is None:
            seg_dir = 'seg'
        if imgFrag is None:
            imgFrag = 'realimage'
        if segFrag is None:
            segFrag = 'encoded_seg'
        img_path = os.path.join(root_dir, img_dir)
        seg_path = os.path.join(root_dir, seg_dir)
        for f in sorted(os.listdir(img_path)):
            file_path = os.path.join(img_path, f)

            #check to see if file is a valid image
            if filetype.is_image(file_path):
                #print(f"{f} is a valid image...")

                #check to see if the segfile exists
                seg_f = f.replace(imgFrag, segFrag).replace(imgfix, segfix)
                seg_file_path = os.path.join(seg_path, seg_f)
                if os.path.exists(seg_file_path):
                    img = Image.open(file_path).convert('L')
                    images.append(img)
                    seg = Image.open(seg_file_path)
                    seg = np.array(seg)
                    annotations.append(seg)
                    file_name.append(f)
        self.images = images
        self.annotations = annotations
        self.filenames = file_name

    def __getitem__(self, index):
        sample = {"image":self.images[index], "annotation":self.annotations[index],
		 "filename":self.filenames[index]}
        return(sample)

    def __len__(self):
        return len(self.images)

    def train_val_dataset(dataset, val_split=0.30, train_split=0.70, random_state=None, shuffle=True, train_idx=None, val_idx=None):
        if train_idx is None or val_idx is None:
            train_idx, valtest_idx = train_test_split(list(range(len(dataset))), test_size=val_split, train_size=train_split, 
                shuffle=shuffle, random_state=random_state)
            val_idx, test_idx = train_test_split(list(range(len(valtest_idx))), test_size=0.5, train_size=0.5, 
                shuffle=shuffle, random_state=random_state)
        datasets = {}
        datasets['train'] = Subset(dataset, train_idx)

        datasets['validation'] = Subset(dataset, val_idx)
        datasets['test'] = Subset(dataset, test_idx)
        return datasets

class ImageSegmentationDataset(Dataset):
    def __init__(self, dataset, processor, transform=None):
        # Initialize the dataset, processor, and transform variables
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    def __len__(self):
        # Return the number of datapoints
        return len(self.dataset)

    def __getitem__(self, idx):
        # Convert the PIL Image to a NumPy array
        image = np.array(self.dataset[idx]["image"].convert("RGB"))
        #print(image.shape)

        # Get the pixel wise instance id and category id maps
        # of shape (height, width)
        instance_seg = np.array(self.dataset[idx]["annotation"])[..., 1]
        class_id_map = np.array(self.dataset[idx]["annotation"])[..., 0]
        class_labels = np.unique(class_id_map)

        inst2class={}
        for label in class_labels:
            instance_ids = np.unique(instance_seg[class_id_map==label])
            inst2class.update({i:label for i in instance_ids})

        # Apply transforms 
        if self.transform is not None:
            transformed = self.transform(image=image, mask = instance_seg)
            (image, instance_seg) =(transformed["image"], transformed["mask"])

            # Convert from channels last to channels first
            image = image.transpose(2,0,1)

        if class_labels.shape[0] == 1 and class_labels[0] == 0:
            # If the image has no objects then it is skipped
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k:v.squeeze() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros(
                (0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1])
            )
        else:
            # Else use process the image with the segmentation maps
            inputs = self.processor(
                [image],
                [instance_seg],
                instance_id_to_semantic_id=inst2class,
                return_tensors="pt"
            )
            inputs = {
                k:v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()
            }
        # Return the inputs
        return inputs

def create_image_fromdataset(index, data_set, savepath, timestamp):

    print(f"[INFO] Displaying an image from dataset index {index} and its annotation...")

    # Using the index grab the corresponding datapoint
    # from the training dataset
    image = data_set[index]["image"]
    image = np.array(image.convert("RGB"))
    annotation = data_set[index]["annotation"]
    annotation = np.array(annotation)
    # Plot the original image and the annotations
    plt.figure(figsize=(15, 5))
    for plot_index in range(2):
        if plot_index == 0:
            # If plot index is 0 display the original image
            plot_image = image
            title = "Original"
        else:
            # Else plot the annotation maps
            plot_image = annotation[..., plot_index - 1]
            title = ["Class Map (R)", "Instance Map (G)"][plot_index - 1]
        # Plot the image
        plt.subplot(1, 2, plot_index + 1)
        plt.imshow(plot_image)
        plt.title(title)
        plt.axis("off")
    plt.savefig(os.path.join(savepath, f"Image_seg_{index}_{timestamp}.png"), dpi = 300)

def collate_fn(examples):
    # Get the pixel values, pixel mask, mask labels, and class labels
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_mask = torch.stack([example["pixel_mask"] for example in examples])
    mask_labels = [example["mask_labels"] for example in examples]
    class_labels = [example["class_labels"] for example in examples]
    # Return a dictionary of all the collated features
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels
    }

def visualize_instance_seg_mask(mask, label2color):
    
    # Initialize image with zeros with the image resolution
    # of the segmentation mask and 3 channels
    image = np.zeros((mask.shape[0], mask.shape[1], 3))
    # Create labels
    labels = np.unique(mask)
    unknown_color = (219, 52, 235)

    for height in range(image.shape[0]):
        for width in range(image.shape[1]):
            if mask[height, width] not in label2color.keys():
              colormask = unknown_color
              print(f'{mask[height, width]} isnt an index in the colormap you passed. setting the color to {colormask}')
            else:
              colormask = mask[height, width]
            image[height, width, :] = label2color[colormask]
    image = image / 255
    return image

def create_loss_plots(avg_train, avg_val, timestamp, savepath):
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(list(range(len(avg_train))),avg_train)
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Train Loss')

    axs[1].plot(list(range(len(avg_val))),avg_val)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Validation Loss')

    plt.subplots_adjust(hspace =0.5)
    plt.savefig(os.path.join(savepath, f'Train_and_Val_loss_{timestamp}.png'), dpi = 300)


def main():

    check_preprocessed = False
    EPOCHS = 25
    NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
    ID2LABEL = {0:'soil', 1:'bedrock', 2:'sand', 3:'bigrock', 255:'unknown'}
    LABEL2ID = {'soil':0, 'bedrock':1, 'sand':2, 'bigrock':3, 'unknown':255}
    
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

    saveweights_location = os.path.join(SAVEROOT, f'weights_{NOW}')
    saveresults_location = os.path.join(SAVEROOT, f'results_{NOW}')

    #make locations if they dont already exist
    os.makedirs(saveweights_location, exist_ok=True)
    os.makedirs(saveresults_location, exist_ok=True)


    # Define the configurations of the transforms specific
    # to the dataset used.

    # this is the mean and std for the AI4MARS dataset.
    ADE_MEAN = np.array([58.514, 58.514, 58.514]) / 255
    ADE_STD = np.array([3.838, 3.838, 3.838]) / 255
    
    #create dataset
    tf_dataset = TransformerDataSet(DATADIR, IMGDIR, SEGDIR, IFRAG, SFRAG, IMFIX, LABFIX)

    training_val_dataset = tf_dataset.train_val_dataset()
    train = training_val_dataset["train"]
    validation = training_val_dataset["validation"]
    test = training_val_dataset["test"]

    print(len(train))
    print(len(validation))
    print(len(test))

    # Create the MaskFormer Image Preprocessor
    processor = MaskFormerImageProcessor(
        do_reduce_labels=True,
        size=(IMAGE_SIZE_X, IMAGE_SIZE_Y),
        ignore_index=255,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
    )

    # Get the MaskFormer config and print it
    config = MaskFormerConfig.from_pretrained(MODEL_NAME)
    print("[INFO] displaying the MaskFormer configuration...")

    # Edit MaskFormer config labels
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    print(config)

    # Use the config object to initialize a MaskFormer model with randomized weights
    model = MaskFormerForInstanceSegmentation(config)

    # Replace the randomly initialized model with the pre-trained model weights
    base_model = MaskFormerModel.from_pretrained(MODEL_NAME)
    model.model = base_model

    # Build the augmentation transforms
    train_val_transform = A.Compose([
        A.Resize(width=IMAGE_SIZE_X, height=IMAGE_SIZE_Y),
        A.HorizontalFlip(p=0.3),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])

    # Build the train and validation instance segmentation dataset
    train_dataset = ImageSegmentationDataset(
        train,
        processor=processor,
        transform=train_val_transform
    )
    val_dataset = ImageSegmentationDataset(
        validation,
        processor=processor,
        transform=train_val_transform
    )

    if check_preprocessed:
        # Check if everything is preprocessed correctly
        inputs = val_dataset[0]
        for k,v in inputs.items():
            print("[INFO] Displaying an shape of the preprocessed inputs...")
            print(k, v.shape)

        for k,v in inputs.items():
            print("[INFO] Displaying an arrays of the preprocessed inputs...")
            print(k, v)


    # Building the training and validation dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Use GPU if available
    print(f'Is cuda available? {torch.cuda.is_available()}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Set number of epochs and batch size
    avg_train = []
    avg_val = []
    num_epochs = EPOCHS
    for epoch in range(num_epochs):
        print(f"Epoch {epoch} | Training")
        # Set model in training mode
        model.train()
        train_loss, val_loss = [], []
        # Training loop
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # Reset the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )
            # Backward propagation
            loss = outputs.loss
            train_loss.append(loss.item())
            loss.backward()
            if idx % 50 == 0:
                print("  Training loss: ", round(sum(train_loss)/len(train_loss), 6))
            # Optimization
            optimizer.step()
        # Average train epoch loss
        avg_train_loss = sum(train_loss)/len(train_loss)
        avg_train.append(avg_train_loss)

        # Set model in evaluation mode
        model.eval()
        print(f"Epoch {epoch} | Validation")
        for idx, batch in enumerate(tqdm(val_dataloader)):
            with torch.no_grad():
                # Forward pass
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                    class_labels=[labels.to(device) for labels in batch["class_labels"]],
                )
                # Get validation loss
                loss = outputs.loss
                val_loss.append(loss.item())
                if idx % 50 == 0:
                    print("  Validation loss: ", round(sum(val_loss)/len(val_loss), 6))
        # Average validation epoch loss
        avg_val_loss = sum(val_loss)/len(val_loss)
        avg_val.append(avg_val_loss)
        # Print epoch losses
        print(f"Epoch {epoch} | train_loss: {avg_train_loss} | validation_loss: {avg_val_loss}")

    # create the loss plots
    create_loss_plots(avg_train, avg_val, NOW, saveresults_location)

    # save the model and preprocessor weights for this session
    model.save_pretrained(saveweights_location)

    # We won't be using albumentations to preprocess images for inference
    processor.do_normalize = True
    processor.do_resize = True
    processor.do_rescale = True

    processor.save_pretrained(saveweights_location)

    # Use random test image
    index = random.randint(0, len(test)-1)
    image = test[index]["image"].convert("RGB")
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
    groundTruthAnn = np.array(test[index]["annotation"])[:,:,0]-1
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

        plt.subplot(1, 3, plot_index+1)
        plt.imshow(plot_image)
        plt.title(title)
        plt.savefig(os.path.join(saveresults_location, f"trainingresult_{NOW}.png"), dpi = 300, bbox_inches='tight')
        #plt.axis("off")

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
        annotation = np.array(validation[idx]["annotation"], dtype='uint16')[:,:,0]
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


if __name__ == "__main__":
    main()
