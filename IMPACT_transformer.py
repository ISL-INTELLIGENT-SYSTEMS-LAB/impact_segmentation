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
#   Define a colormap for our semantic segmentation files
#   index, color, name
#   ____________________

#   1, (0, 0, 0), "bedrock"
#   2, (42,114,60), "sky"
#   3, (89, 89, 89), "sand/ground"
#   4, (255, 0, 0), "ventifact"
#   5, (0,255,0), "science_rock"
#   6, (0, 0, 255), "blueberry"
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
#   None
#
#  FUNCTIONS
#  ---------
#   flip_colors
#   create_image_mask_dictionary
#   encodeImageMask
#   mapImageMask
#   createEncoding
#
'''
#################################################################################################
#Import Statements
#################################################################################################
import os
import sys

# extremely important that you change these variables for your particular setup.
VIR_ENV_DIR = 'vir_env'
VIR_ENV_NAME = 'simulation'
HOME = os.path.join('/home',os.getlogin())
# I put my virtual environment in /home/cspooner/vir_env/simulation

# For use on the cluster only. 
if sys.platform == 'linux':
    
    SIM_VIRENV_LIBS = os.path.join(HOME, VIR_ENV_DIR, VIR_ENV_NAME, 'lib','python3.7', 'site-packages')
    print(SIM_VIRENV_LIBS)

    if SIM_VIRENV_LIBS not in sys.path:
        sys.path.insert(0,SIM_VIRENV_LIBS)

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
    def __init__(self, root_dir, img_dir=None, seg_dir=None, imgFrag=None, segFrag=None):
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
                seg_f = f.replace(imgFrag, segFrag)
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

    def train_val_dataset(dataset, val_split=0.30, random_state=None, shuffle=True, train_idx=None, val_idx=None):
        if train_idx is None or val_idx is None:
            train_idx, valtest_idx = train_test_split(list(range(len(dataset))), test_size=val_split, shuffle=shuffle, random_state=random_state)
            print(len(valtest_idx))
            val_idx, test_idx = train_test_split(list(range(len(valtest_idx))), test_size=0.5, shuffle=shuffle, random_state=random_state)
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

def visualize_instance_seg_mask(mask):
    # Initialize image with zeros with the image resolution
    # of the segmentation mask and 3 channels
    image = np.zeros((mask.shape[0], mask.shape[1], 3))
    # Create labels
    labels = np.unique(mask)
    label2color = {
        label: (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        for label in labels
    }
    for height in range(image.shape[0]):
        for width in range(image.shape[1]):
            image[height, width, :] = label2color[mask[height, width]]
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
    EPOCHS = 1
    NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
    ID2LABEL = {0:'bedrock', 1:'sky', 2:'sand/ground', 3:'ventifact', 4:'science_rock', 5:'blueberry'}
    LABEL2ID = {'bedrock':0, 'sky':1, 'sand':2, 'ventifact':3, 'science_rock':4, 'blueberry':5}

    PROJECTDIR = 'simulations'
    ROOTDIR = os.path.join(HOME, PROJECTDIR, 'Data_Collection','test_dataset')
    IMGDIR = 'realimage'
    IFRAG = 'realimage'
    SEGDIR = 'seg'
    SFRAG = 'encoded_seg'

    # Define the name of the mode
    MODEL_NAME = "facebook/maskformer-swin-base-ade"
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 4

    saveweights_location = os.path.join(HOME, PROJECTDIR,f'weights_{NOW}')
    saveresults_location = os.path.join(HOME, PROJECTDIR,f'results_{NOW}')

    #make locations if they dont already exist
    os.makedirs(saveweights_location, exist_ok=True)
    os.makedirs(saveresults_location, exist_ok=True)

    ###
    # NEED TO GET THE MEAN AND STD OF DATASET
    ###

    # Define the configurations of the transforms specific
    # to the dataset used
    ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

    #create dataset
    tf_dataset = TransformerDataSet(ROOTDIR, IMGDIR, SEGDIR, IFRAG, SFRAG)

    training_val_dataset = tf_dataset.train_val_dataset()
    train = training_val_dataset["train"]
    validation = training_val_dataset["validation"]
    test = training_val_dataset["test"]

    # Create the MaskFormer Image Preprocessor
    processor = MaskFormerImageProcessor(
        do_reduce_labels=True,
        size=(512, 512),
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
        A.Resize(width=512, height=512),
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
    print(torch.cuda.is_available())
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

    instance_seg_mask_disp = visualize_instance_seg_mask(instance_seg_mask)
    plt.figure(figsize=(10, 10))
    for plot_index in range(2):
        if plot_index == 0:
            plot_image = image
            title = "Original"
        else:
            plot_image = instance_seg_mask_disp
            title = "Segmentation"

        plt.subplot(1, 2, plot_index+1)
        plt.imshow(plot_image)
        plt.title(title)
        plt.savefig(os.path.join(saveresults_location, f"trainingresult_{NOW}.png"), dpi = 300)
        plt.axis("off")

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
    results = metrics.compute(
        predictions=preds,
        references=ground_truths,
        num_labels=100,
        ignore_index=255
    )
    print(f"Mean IoU: {results['mean_iou']} | Mean Accuracy: {results['mean_accuracy']} | Overall Accuracy: {results['overall_accuracy']}")


if __name__ == "__main__":
    main()
