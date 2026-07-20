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
#import matplotlib.cm as cm
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from typing import Tuple
import kornia as K
import kornia.feature as KF

# Import custom modules for transforms, plotting, and models
from kornia_moons.viz import draw_LAF_matches
from pathlib import Path
from shapely.geometry import Point, Polygon
from torchvision.ops import box_convert
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
#from groundingdino.util.inference import load_model, load_image, predict, annotate
from sam2.sam2_image_predictor import SAM2ImagePredictor





# =============================================================================
# Helper Functions
# =============================================================================



# Function to handle mouse click event
def get_pixel_location(event, x, y, flags, param):
    pixel_location = []
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
        pixel_location.append((x, y))
        print(f"Pixel location: ({x}, {y})")
        #cv2.destroyAllWindows()  # Close the window after clicking
    return pixel_location

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
    
# Create an output directory with a unique name based on the current date and time.
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
# Load and Process the Input Image Pair
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

# Now let’s clean-up the correspondences with modern RANSAC 
# and estimate fundamental matrix between two images
mkpts0 = correspondences["keypoints0"].cpu().numpy()
mkpts1 = correspondences["keypoints1"].cpu().numpy()
Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
inliers = inliers > 0

'''
# draw all matches
draw_LAF_matches(
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts0).view(1, -1, 2),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1),
    ),
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts1).view(1, -1, 2),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1),
    ),
    torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
    K.tensor_to_image(img0),
    K.tensor_to_image(img1),
    inliers,
    draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 1, 1), "vertical": False},
)
#plt.show()
'''

'''
# Create a color map for visualizing match confidence
color0 = cm.jet(mconf, alpha=0.7)
# Create a text label that will appear on the matching figure
text0 = ['LoFTR', f'Matches: {len(mkpts0)}']


# Save a high-resolution PDF of the matching figure using a custom plotting function.
matching_name = f"{LOFTR_MATCHING_NAME}_whole_image.pdf"
make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color0,
                     mkpts0, mkpts1, text0, path=os.path.join(OUTPUT_DIR, matching_name))
'''


# Load the image
image = cv2.imread(IMG0_PTH)

# Display the image
cv2.imshow('Click on the image', image)

# Set the mouse callback function to capture clicks
cv2.setMouseCallback('Click on the image', get_pixel_location)

print("Press 'q' to quit.")
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
# =============================================================================
# GroundingDINO Object Detection 
# =============================================================================

gdinoprocessor = AutoProcessor.from_pretrained(GDINO_MODEL_CFG)
gdinomodel = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_CFG).to(DEVICE)

# Load and preprocess the first image (IMG0) for object detection. Needs to resized to match
# the loftr image sizes
img0_pil = Image.open(IMG0_PTH)
img0_pil = img0_pil.resize((RESIZE_WIDTH, RESIZE_HEIGHT ))

inputs = gdinoprocessor(images=img0_pil, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    outputs = gdinomodel(**inputs)

results = gdinoprocessor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    target_sizes=[img0_pil.size[::-1]]
)

gdino_boxes = results[0]['boxes'].cpu()
gdino_labels = results[0]['labels']

# =============================================================================
# SAM2 Mask Generation
# =============================================================================

predictor = SAM2ImagePredictor.from_pretrained(SAM_MODEL_CFG)

# =============================================================================
# Loop Over Each Detected Object Box to Refine Matches and Generate Masks
# =============================================================================

sam_masks = []
sam_scores = []

for i, box in enumerate(gdino_boxes):
  # Generate a mask for the current detected box using SAM2.
  
  # Set the SAM2 predictor to use the processed image.
  with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(img0_pil)
        
    masks0, scores0, logits0 = predictor.predict(
      point_coords=None,      # No initial points provided
      point_labels=None,      # No initial point labels
      box=box,       # Provide the current box for mask generation
      multimask_output=False, # Generate a single mask output
    )
    sam_masks.append(masks0)
    sam_scores.append(scores0[0])
    

    h, w = img0_pil.size
    
    # Flatten the mask to create a binary image and extract its contours.
    #
    flat_mask = masks0.flatten().reshape(h, w)

    contours, hierarchy = cv2.findContours(flat_mask.astype('uint8'),
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    #print(contours[0])
    shapely_polygons = []
    for c in contours:

      if c.ndim == 3:
        c = c[:, 0, :]  # Flatten if contour is in (N, 1, 2) format
        #print(c)
        shapely_polygons.append(c)
    
    #polygon = Polygon(shapely_polygons)
    '''
    #shape_contours = [x[0] for x in contours[0].tolist()]
    #print(shape_contours)
    # Create a polygon from the contour points if there are enough points to form one.
    shape_poly = Polygon(shape_contours) if len(shape_contours) >= 4 else None

    print(f'Do we have any points in common in the picture? : {shape_poly}')
    
    if shape_poly is not None:
        # Initialize lists to store refined matching points within the detected object.
        picture0points = []
        picture1points = []
        matchconf = []
        points1labels = []
        
        # Iterate through all LoFTR keypoints in the first image.
        for idx, point in enumerate(mkpts0):
            # Create a shapely Point from the keypoint.
            if Point(point).within(shape_poly):
                # If the point lies within the detected object's polygon, store it.
                picture0points.append(mkpts0[idx])
                picture1points.append(mkpts1[idx])
                points1labels.append(1)
                matchconf.append(mconf[idx])

        # Convert the collected points to numpy arrays for plotting and further processing.
        picture0points = np.array(picture0points)
        picture1points = np.array(picture1points)
        points1labels = np.array(points1labels)

        # Create a color map for the refined matches.
        color1 = cm.jet(matchconf, alpha=0.7)
        text1 = ['LoFTR', f'Matches: {len(picture0points)}']

        
        try:
            # Save a matching figure for the detected object region.
            object_matching_name = f"{LOFTR_MATCHING_NAME}_{phrases[i]}.pdf"
            make_matching_figure(img0_raw, img1_raw, picture0points, picture1points,
                                 color1, picture0points, picture1points,
                                 text1, path=os.path.join(OUTPUT_DIR, object_matching_name))

            # Load and preprocess the second image (IMG1) for generating a corresponding mask.
            image_source1, image1 = load_image_fsu(IMG1_PTH, RESIZE_WIDTH, RESIZE_HEIGHT)
            h1, w1, _ = image_source1.shape
            pil_image1 = Image.open(IMG1_PTH).resize((w1, h1))
            pil_image1 = np.array(pil_image1.convert("RGB"))
            predictor.set_image(pil_image1)

            # Generate a mask for the second image using the refined keypoints as guidance.
            masks1, scores1, _ = predictor.predict(
                point_coords=picture1points,
                point_labels=points1labels,
                multimask_output=False,
            )

            # Display and save the generated mask for the second image.
            show_masks(pil_image1, masks1, scores1, randomColor=True)
            image1_maskname = f'{LOFTR_MATCHING_NAME}_{os.path.basename(IMG1_PTH).split(".")[0]}_{phrases[i]}_{i}.jpg'
            plt.savefig(os.path.join(OUTPUT_DIR, image1_maskname), dpi=300, bbox_inches='tight')

        except Exception as e:
            # If any errors occur during processing, print a warning message with details.
            print(f"Warning: there might not be any matches for {phrases[i]}.")
            print(f"Exception: {e}")

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
            plt.show()'''

# =============================================================================
# Display the generated masks (using gdino bboxs and SAM2) 
# on image 0 with random colors.
# =============================================================================

show_masks_new(img0_pil, sam_masks, sam_scores)
plt.show()

image0_maskname = f'gdino_localization_img0_{os.path.basename(IMG0_PTH).split(".")[0]}.jpg'
plt.savefig(os.path.join(OUTPUT_DIR, image0_maskname), dpi=300, bbox_inches='tight')

# =============================================================================
# Display the generated masks (using gdino bboxs and SAM2) 
# on image 1 with random colors.
# =============================================================================
'''
show_masks_new(img0_pil, sam_masks, sam_scores)
#plt.show()

image0_maskname = f'gdino_localization_img0_{os.path.basename(IMG0_PTH).split(".")[0]}.jpg'
plt.savefig(os.path.join(OUTPUT_DIR, image0_maskname), dpi=300, bbox_inches='tight')
'''