"""
LoFTR Scene Matching

This script demonstrates how to run the LoFTR matching algorithm on a pair of images,
followed by object detection (using GroundingDINO) and mask generation (using SAM2).
The configuration constants (such as file paths, thresholds, and image sizes) are
loaded from an external YAML file.

*NOTE:  LofTR works on grayscale images only

Before running, ensure that:
  - A 'loftr_config.yaml' file with the required settings exists in the same directory.
  - The necessary modules and dependencies are installed.
"""

import os
import cv2
import torch
import yaml
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from typing import Tuple
import kornia as K
import kornia.feature as KF
import pycocotools.mask as mask_util
from matplotlib.patches import ConnectionPatch

# Import custom modules for transforms, plotting, and models
from kornia_moons.viz import draw_LAF_matches
from pathlib import Path
from shapely.geometry import Point, Polygon
from torchvision.ops import box_convert
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
#from groundingdino.util.inference import load_model, load_image, predict, annotate
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# =============================================================================
# Helper Functions
# =============================================================================
def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)
    
def show_mask(mask, ax, random_color=False, borders=True):
    """
    Display a segmentation mask on the provided matplotlib axis.

    Parameters:
      mask (numpy.ndarray): Binary mask array.
      ax (matplotlib.axes.Axes): Axis to display the mask.
      random_color (bool): If True, use a random color; otherwise use a fixed color.
      borders (bool): If True, draw borders around the mask regions.
    """
    # Set color: random or fixed (using RGB values scaled to [0, 1])
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    # Get dimensions of the mask and convert the mask type to unsigned 8-bit
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    
    # Multiply mask by the color to create an RGBA image for overlaying
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    # If borders are enabled, compute contours and draw them
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Approximate contours to smooth them out
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        # Draw contours on the mask image with a white border (with transparency)
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    
    # Display the mask overlay on the provided axis
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    """
    Plot points on the image: positive points in green and negative in red.

    Parameters:
      coords (numpy.ndarray): Coordinates of the points.
      labels (numpy.ndarray): Binary labels corresponding to each point (1 for positive, 0 for negative).
      ax (matplotlib.axes.Axes): Axis to plot the points.
      marker_size (int): Size of the plotted markers.
    """
    # Separate points by label
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    
    # Plot positive points (green stars) and negative points (red stars)
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
               s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    """
    Draw a bounding box on the given axis.

    Parameters:
      box (list or numpy.ndarray): Box coordinates in [x_min, y_min, x_max, y_max] format.
      ax (matplotlib.axes.Axes): Axis to draw the box.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    # Draw a rectangle with a green edge and transparent fill
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(image, masks, scores, point_coords=None, box_coords=None,
               input_labels=None, borders=True, randomColor=False):
    """
    Display multiple masks on an image, optionally overlaying point and box annotations.

    Parameters:
      image (numpy.ndarray): The image on which to display masks.
      masks (list): List of mask arrays.
      scores (list): List of confidence scores corresponding to each mask.
      point_coords (numpy.ndarray): Coordinates for point annotations (optional).
      box_coords (list): Box coordinates for additional annotations (optional).
      input_labels (numpy.ndarray): Labels for the point coordinates (optional).
      borders (bool): If True, draw borders around the masks.
      randomColor (bool): If True, use random colors for each mask.
    """
    # Loop through each mask and display it along with optional annotations
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders, random_color=randomColor)
        if point_coords is not None:
            # Ensure that labels for the points are provided if points are to be shown
            assert input_labels is not None, "Point labels must be provided when showing points."
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        # If there are multiple masks, add a title to distinguish them
        #if len(scores) > 1:
        #    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')

def show_masks_new(image, masks, scores, random_color=True):
    """
    Display multiple masks on an image, optionally overlaying point and box annotations.

    Parameters:
      image (numpy.ndarray): The image on which to display masks.
      masks (list): List of mask arrays.
      scores (list): List of confidence scores corresponding to each mask.
      point_coords (numpy.ndarray): Coordinates for point annotations (optional).
      box_coords (list): Box coordinates for additional annotations (optional).
      input_labels (numpy.ndarray): Labels for the point coordinates (optional).
      borders (bool): If True, draw borders around the masks.
      randomColor (bool): If True, use random colors for each mask.
    """
    plt.imshow(image)
    # Loop through each mask and display it along with optional annotations
    for i, (mask, score) in enumerate(zip(masks, scores)):
      if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
      else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
        
      h, w = mask.shape[-2:]
      mask = mask.astype(np.uint8)
      
      mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
      
      contours, _ = cv2.findContours(mask[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
      # Draw contours on the mask image with a white border (with transparency)
      mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.8), thickness=2)
        
      
      plt.imshow(mask_image, alpha=0.8)

    plt.axis('off')

    
def _DFS(polygons, contours, hierarchy, sibling_id, is_outer, siblings):
  while sibling_id != -1:
    contour = contours[sibling_id].squeeze(axis=1)
    if len(contour) >= 3:
      first_child_id = hierarchy[sibling_id][2]
      children = [] if is_outer else None
      _DFS(polygons, contours, hierarchy, first_child_id, not is_outer, children)

      if is_outer:
        polygon = Polygon(contour, holes=children)
        polygons.append(polygon)
      else:
        siblings.append(contour)

    sibling_id = hierarchy[sibling_id][0] 
    
def generate_polygons(contours, hierarchy):
  """Generates a list of Shapely polygons from the contours hirarchy returned by cv2.find_contours().
     The list of polygons is generated by performing a depth-first search on the contours hierarchy tree.
  Parameters
  ----------
  contours : list
    The contours returned by cv2.find_contours()
  hierarchy : list
    The hierarchy returned by cv2.find_contours()
  Returns
  -------
  list
    The list of generated Shapely polygons
  """
  
  hierarchy = hierarchy[0]
  polygons = []
  _DFS(polygons, contours, hierarchy, 0, True, [])
  return polygons
  
def draw_matches(img0, img1, matchpoints0, matchpoints1, matchconf):
  alpha = 0.1
  # Create a color map for visualizing match confidence
  colors = cm.jet(matchconf)
  fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
  axes[0].imshow(img0)
  axes[0].scatter(matchpoints0[:, 0], matchpoints0[:, 1], marker=".", color=colors, s=5, alpha=alpha)
  axes[0].axis('off')
  axes[1].imshow(img1)
  axes[1].scatter(matchpoints1[:, 0], matchpoints1[:, 1], marker=".", color=colors, s=5, alpha=alpha)
  axes[1].axis('off')
  for i, x in enumerate(matchpoints0):
    start = (int(matchpoints0[i][0]),int(matchpoints0[i][1]))
    end = (int(matchpoints1[i][0]), int(matchpoints1[i][1]))

    con = ConnectionPatch(xyA=start, xyB=end, coordsA="data", coordsB="data",
                        axesA=axes[0], axesB=axes[1], color=colors[i], alpha=alpha)
    axes[1].add_artist(con)
  return fig
  #plt.show()

# =============================================================================
# Load Configuration from YAML File
# =============================================================================

# Open and parse the YAML configuration file.
with open("loftr_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Assign configuration parameters to variables
EXPERIMENT_NAME = config["EXPERIMENT_NAME"]
IMG0_PTH = Path(config["IMG0_PTH"])
IMG1_PTH = Path(config["IMG1_PTH"])
GDINO_MODEL_CFG = config["GDINO_MODEL_ID"]
TEXT_PROMPT = config["TEXT_PROMPT"]
BOX_THRESHOLD = config["BOX_THRESHOLD"]
TEXT_THRESHOLD = config["TEXT_THRESHOLD"]
LOFTR_IMAGE_TYPE = config["LOFTR_IMAGE_TYPE"]
SAM_MODEL_CFG = config["SAM2_MODEL_ID"]
DISPLAY_POINTS_INSIDE = config["DISPLAY_POINTS_INSIDE"]
LOFTR_MATCHING_NAME = config["LOFTR_MATCHING_NAME"]
RESIZE_HEIGHT = config["RESIZE_HEIGHT"]
RESIZE_WIDTH = config["RESIZE_WIDTH"]

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
    
# Create an output directory with a unique name based on the 
# experiment name as well as the current date and time.
now = datetime.now()
date = now.strftime("%Y%m%d")
time = now.strftime("%H%M")
OUTPUT_DIR = os.path.join(EXPERIMENT_NAME, date, time)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Create a label map 
# =============================================================================

labelDict = {}
text_prompt_list = TEXT_PROMPT.split(" . ")
for j, element in enumerate(text_prompt_list):
  if element not in labelDict:
    labelDict[element] = j
#print(labelDict)

# =============================================================================
# Load and Process the Input Image Pair using kornia
# =============================================================================

# Read the first image and resize
img0 = K.io.load_image(IMG0_PTH, K.io.ImageLoadType.RGB32)[None, ...]
img0 = K.geometry.resize(img0, (RESIZE_HEIGHT, RESIZE_WIDTH), antialias=True)

# Read the second image and resize
img1 = K.io.load_image(IMG1_PTH, K.io.ImageLoadType.RGB32)[None, ...]
img1 = K.geometry.resize(img1, (RESIZE_HEIGHT, RESIZE_WIDTH), antialias=True)


# =============================================================================
# Run LoFTR Inference and Save the Matching Figure
# =============================================================================

matcher = KF.LoFTR(pretrained=LOFTR_IMAGE_TYPE)

input_dict = {
    "image0": K.color.rgb_to_grayscale(img0),  # LofTR works on grayscale images only
    "image1": K.color.rgb_to_grayscale(img1),
}

with torch.inference_mode():
    correspondences = matcher(input_dict)

mkpts0 = correspondences["keypoints0"].cpu().numpy()
mkpts1 = correspondences["keypoints1"].cpu().numpy()
mconf = correspondences["confidence"].cpu().numpy()

# =============================================================================
# Load the Input Image Pair using pillow, and resize to same size used in LoFTR
# =============================================================================

img0_pil = Image.open(IMG0_PTH).convert("RGB")
img0_pil = img0_pil.resize((RESIZE_WIDTH, RESIZE_HEIGHT ))

img1_pil = Image.open(IMG1_PTH).convert("RGB")
img1_pil = img1_pil.resize((RESIZE_WIDTH, RESIZE_HEIGHT ))

loftrmatches = draw_matches(img0_pil, img1_pil, mkpts0, mkpts1, mconf)

matching_name = f"ALL_{LOFTR_MATCHING_NAME}_whole_image.png"

plt.savefig(os.path.join(OUTPUT_DIR, matching_name), dpi=300, bbox_inches="tight")
#plt.show()
plt.clf()
'''
# =============================================================================
# SAM2 Mask Generation
# =============================================================================
# SAM2 predictor given keypoints, boxes
predictor = SAM2ImagePredictor.from_pretrained(SAM_MODEL_CFG)

# SAM2 mask generator assuming nothing. creates all masks
mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(SAM_MODEL_CFG)


img0_np = np.array(img0_pil.convert("RGB"))

# select the device for computation

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
masks0 = mask_generator.generate(img0_np)

sam_masks1 = []
sam_scores1 = []

for mask in masks0:        
  # squeeze the mask to remove extra dimension, then cast to int to create a binary image and extract its contours.
  #
  flat_mask = np.squeeze(mask['segmentation']).astype('uint8')

  contours, hierarchy = cv2.findContours(flat_mask.astype('uint8'),
                                         cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_SIMPLE)
                                         

  polygons = generate_polygons(contours, hierarchy)

  if len(polygons) == 1:
      # Initialize lists to store refined matching points within the detected object.
      picture0points = []
      picture1points = []
      matchconf = []
      points1labels = []
      
      # Iterate through all LoFTR keypoints in the first image.
      for idx, point in enumerate(mkpts0):
          # Create a shapely Point from the keypoint.

          if Point(point).within(polygons):
              print(mkpts0[idx])
              # If the point lies within the detected object's polygon, store it.
              picture0points.append(mkpts0[idx])
              picture1points.append(mkpts1[idx])
              points1labels.append(1)
              matchconf.append(mconf[idx])
      print(len(picture0points))
      # Convert the collected points to numpy arrays for plotting and further processing.
      picture0points = np.array(picture0points)
      picture1points = np.array(picture1points)
      points1labels = np.array(points1labels)      
      
      if len(picture0points) > 0:
          # Save a matching figure for the detected object region.
          #object_matching_name = f"{LOFTR_MATCHING_NAME}_{phrases[i]}.pdf"
          #make_matching_figure(img0_raw, img1_raw, picture0points, picture1points,
                               #color1, picture0points, picture1points,
                               #text1, path=os.path.join(OUTPUT_DIR, object_matching_name))

          # Load and preprocess the second image (IMG1) for generating a corresponding mask.
      
          # Set the SAM2 predictor to use the processed image.

          with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(img1_pil)
            
            masks1, scores1, logits1 = predictor.predict(
              point_coords=picture1points,      # No initial points provided
              point_labels=points1labels,      # No initial point labels
              multimask_output=False, # Generate a single mask output
              )
            sam_masks1.append(masks1)
            sam_scores1.append(scores1[0])
            
      else:
          # If any errors occur during processing, print a warning message with details.
          print("This polygon does not appear to have any loftr matches")

      # Optionally, display the matching points along with the detected object's contour.
      if DISPLAY_POINTS_INSIDE:
          plt.figure(figsize=(10, 10))
          plt.imshow(img0_rgb)
          # Plot the original LoFTR keypoints in red.
          plt.scatter(mkpts0[:, 0], mkpts0[:, 1], marker="x", color="red", s=200)
          # Transpose the contour points for plotting.
          shape_contours_T = np.transpose(shape_contours)
          plt.scatter(shape_contours_T[0], shape_contours_T[1], marker=".", color="blue", s=200)
          plt.savefig(os.path.join(OUTPUT_DIR, "pointsInside.jpg"), dpi=300)
          plt.clf()
          #plt.show()

# =============================================================================
# Display the generated masks (using gdino bboxs and SAM2) 
# on image 0 with random colors.
# =============================================================================
plt.imshow(img0_pil)
show_anns(masks0)
#show_masks_new(img0_pil, masks)

image0_maskname = f'allobjects_localization_img0_{os.path.basename(IMG0_PTH).split(".")[0]}.jpg'
plt.axis('off')
plt.savefig(os.path.join(OUTPUT_DIR, image0_maskname), dpi=300, bbox_inches='tight')
#plt.show()
plt.clf()

# =============================================================================
# Display the generated masks (using gdino bboxs and SAM2) 
# on image 1 with random colors.
# =============================================================================

show_masks_new(img1_pil, sam_masks1, sam_scores1)

image1_maskname = f'matched_objects_img1_{os.path.basename(IMG1_PTH).split(".")[0]}.jpg'
plt.axis('off')
plt.savefig(os.path.join(OUTPUT_DIR, image1_maskname), dpi=300, bbox_inches='tight')
#plt.show()
plt.clf()'''