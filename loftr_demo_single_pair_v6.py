"""
LoFTR Demo with Custom Image Pairs

This script demonstrates how to run the LoFTR matching algorithm on a pair of images,
followed by object detection (using GroundingDINO) and mask generation (using SAM2).
The configuration constants (such as file paths, thresholds, and image sizes) are
loaded from an external YAML file.

Before running, ensure that:
  - A 'config.yaml' file with the required settings exists in the same directory.
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

# Import custom modules for transforms, plotting, and models
import groundingdino.datasets.transforms as T
from software_for_scenes.LoFTR.src.utils.plotting import make_matching_figure
from software_for_scenes.LoFTR.src.loftr import LoFTR, default_cfg
from shapely.geometry import Point, Polygon
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict, annotate
from software_for_scenes.sam2.sam2.build_sam import build_sam2
from software_for_scenes.sam2.sam2.sam2_image_predictor import SAM2ImagePredictor

# =============================================================================
# Helper Functions
# =============================================================================

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
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')

def load_image_fsu(image_path: str, w: int, h: int) -> Tuple[np.array, torch.Tensor]:
    """
    Load an image from the given path, resize it, and apply a transformation pipeline.

    Parameters:
      image_path (str): Path to the input image.
      w (int): Desired width of the output image.
      h (int): Desired height of the output image.

    Returns:
      image (numpy.ndarray): The resized image as a NumPy array.
      image_transformed (torch.Tensor): The transformed image tensor (normalized, etc.).
    """
    # Define a transformation pipeline including resizing, tensor conversion, and normalization.
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Open the image and convert it to RGB format
    image_source = Image.open(image_path).convert("RGB")
    # Resize the image to the desired dimensions
    image_source = image_source.resize((w, h))
    # Convert the image to a numpy array for display or further processing
    image = np.asarray(image_source)
    # Apply the transformation pipeline
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

# =============================================================================
# Load Configuration from YAML File
# =============================================================================

# Open and parse the YAML configuration file.
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Assign configuration parameters to variables
IMG0_PTH = config["IMG0_PTH"]
IMG1_PTH = config["IMG1_PTH"]
IMAGE_TYPE = config["IMAGE_TYPE"]
TEXT_PROMPT = config["TEXT_PROMPT"]
BOX_THRESHOLD = config["BOX_THRESHOLD"]
TEXT_THRESHOLD = config["TEXT_THRESHOLD"]
LOFTR_INDOOR_PATH = config["LOFTR_INDOOR_PATH"]
LOFTR_OUTDOOR_PATH = config["LOFTR_OUTDOOR_PATH"]
GDINO_SCRIPT = config["GDINO_SCRIPT"]
GDINO_WEIGHTS = config["GDINO_WEIGHTS"]
SAM_CHECKPOINT = config["SAM_CHECKPOINT"]
SAM_MODEL_CFG = config["SAM_MODEL_CFG"]
DISPLAY_POINTS_INSIDE = config["DISPLAY_POINTS_INSIDE"]
LOFTR_MATCHING_NAME = config["LOFTR_MATCHING_NAME"]
RESIZE_HEIGHT = config["RESIZE_HEIGHT"]
RESIZE_WIDTH = config["RESIZE_WIDTH"]

# Create an output directory with a unique name based on the current date and time.
OUTPUT_DIR = datetime.now().strftime("%Y%m%d_%H%M")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Load and Process the Input Image Pair
# =============================================================================

# Read the first image in grayscale and in BGR (for color conversion)
img0_raw = cv2.imread(IMG0_PTH, cv2.IMREAD_GRAYSCALE)
img0_bgr = cv2.imread(IMG0_PTH)
# Convert BGR image to RGB for consistent color display
img0_rgb = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
# Resize the RGB image for display and matching
img0_rgb = cv2.resize(img0_rgb, (RESIZE_WIDTH, RESIZE_HEIGHT))

# Read the second image in grayscale
img1_raw = cv2.imread(IMG1_PTH, cv2.IMREAD_GRAYSCALE)
# Resize both images to the specified dimensions
img0_raw = cv2.resize(img0_raw, (RESIZE_WIDTH, RESIZE_HEIGHT))
img1_raw = cv2.resize(img1_raw, (RESIZE_WIDTH, RESIZE_HEIGHT))

# Prepare the images for LoFTR:
# Convert images to tensors, add necessary dimensions, move to GPU, and normalize pixel values
img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}

# =============================================================================
# Run LoFTR Inference and Save the Matching Figure
# =============================================================================

# Initialize the LoFTR matcher using the default configuration
matcher = LoFTR(config=default_cfg)

# Load the appropriate pre-trained model weights based on the image type (indoor or outdoor)
if IMAGE_TYPE == 'indoor':
    state_dict = torch.load(LOFTR_INDOOR_PATH)['state_dict']
elif IMAGE_TYPE == 'outdoor':
    state_dict = torch.load(LOFTR_OUTDOOR_PATH)['state_dict']
else:
    raise ValueError("Wrong image_type is given. Must be either 'indoor' or 'outdoor'.")
matcher.load_state_dict(state_dict)
matcher = matcher.eval().cuda()  # Set matcher to evaluation mode and move to GPU

# Run inference without tracking gradients (saves memory and computation)
with torch.no_grad():
    matcher(batch)
    # Retrieve matching keypoints and their confidence scores from the batch
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()

# Create a color map for visualizing match confidence
color0 = cm.jet(mconf, alpha=0.7)
# Create a text label that will appear on the matching figure
text0 = ['LoFTR', f'Matches: {len(mkpts0)}']

# Save a high-resolution PDF of the matching figure using a custom plotting function.
matching_name = f"{LOFTR_MATCHING_NAME}_whole_image.pdf"
make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color0,
                     mkpts0, mkpts1, text0, path=os.path.join(OUTPUT_DIR, matching_name))

# =============================================================================
# GroundingDINO Object Detection and SAM2 Mask Generation
# =============================================================================

# Load the GroundingDINO model using the specified script and weights.
model = load_model(GDINO_SCRIPT, GDINO_WEIGHTS)

# Load and preprocess the first image (IMG0) for object detection.
image_source0, image0 = load_image_fsu(IMG0_PTH, RESIZE_WIDTH, RESIZE_HEIGHT)

# Run object detection on the image using the provided text prompt.
boxes, logits, phrases = predict(
    model=model,
    image=image0,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)
print(phrases)  # Print the detected object phrases to the console

# Build and initialize the SAM2 model for mask generation.
sam2_model = build_sam2(SAM_MODEL_CFG, SAM_CHECKPOINT, device=torch.device("cuda"))
predictor = SAM2ImagePredictor(sam2_model)

# Open the first image using PIL, resize it to match the processed image dimensions,
# and convert it to an RGB numpy array.
pil_image0 = Image.open(IMG0_PTH)
h, w, _ = image_source0.shape
pil_image0 = pil_image0.resize((w, h))
pil_image0 = np.array(pil_image0.convert("RGB"))
# Set the SAM2 predictor to use the processed image.
predictor.set_image(pil_image0)

# Convert the detected boxes (which are relative coordinates) to absolute pixel coordinates.
boxes_cxcywh = boxes * torch.Tensor([w, h, w, h])
boxes_xyxy = box_convert(boxes=boxes_cxcywh, in_fmt="cxcywh", out_fmt="xyxy").numpy()

# =============================================================================
# Loop Over Each Detected Object Box to Refine Matches and Generate Masks
# =============================================================================

for i, box in enumerate(boxes_xyxy):
    # Generate a mask for the current detected box using SAM2.
    masks0, scores0, _ = predictor.predict(
        point_coords=None,      # No initial points provided
        point_labels=None,      # No initial point labels
        box=box[None, :],       # Provide the current box for mask generation
        multimask_output=False, # Generate a single mask output
    )
    
    # Display the generated masks on the image with random colors.
    show_masks(pil_image0, masks0, scores0, randomColor=True)
    # Save the mask image with a unique name based on the phrase and index.
    image0_maskname = f'{LOFTR_MATCHING_NAME}_{os.path.basename(IMG0_PTH).split(".")[0]}_{phrases[i]}_{i}.jpg'
    plt.savefig(os.path.join(OUTPUT_DIR, image0_maskname), dpi=300, bbox_inches='tight')

    # Flatten the mask to create a binary image and extract its contours.
    flat_mask = masks0.flatten().reshape(h, w)
    contours, hierarchy = cv2.findContours(flat_mask.astype('uint8'),
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # Extract the first contour as a list of points.
    shape_contours = [x[0] for x in contours[0].tolist()]
    # Create a polygon from the contour points if there are enough points to form one.
    shape_poly = Polygon(shape_contours) if len(shape_contours) >= 4 else None

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
            plt.show()
