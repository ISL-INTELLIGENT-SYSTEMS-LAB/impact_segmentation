
# Webcam Capture & Analysis Application

## Overview
This application provides a graphical interface to capture images from a webcam, perform background removal to segment the main subject, and compute various image metrics.

## Features
- **Image Capture:**
  - Capture two images from the live webcam feed.
  - Automatically segment the main subject using the `rembg` library.

- **Image Analysis:**
  - Calculate multiple metrics for each captured image, including:
    - Subject centroid
    - Contour area
    - Bounding box
    - Contour perimeter
    - Aspect ratio
    - Average brightness
    - Average color
  - Compute the angle difference between the two images based on the subject's horizontal offset.

- **Visualization:**
  - Generate a circle diagram visualizing:
    - Subject's position (blue dot)
    - Camera positions (red dots)
    - Red lines indicating the calculated angles
    - A legend explaining diagram elements

- **Data Export:**
  - Export data into a timestamped folder containing:
    - Full-resolution captured images
    - Segmentation masks
    - The circle diagram saved as `Camera_pos_diagram.png`
    - A `data.txt` file with detailed computed metrics

- **User Interface:**
  - Scaled interface displaying a resized live video feed and captured images.
  - Automatic window positioning at the top center of the screen.

## Installation
Before running the application, ensure that the required dependencies are installed:

```bash
pip install -r requirements.txt
```

## Requirements
- numpy
- opencv-python
- Pillow
- rembg
- onnxruntime

## Usage
1. Run the Python script to launch the application:

```bash
python webcam_capture_analysis.py
```

2. Use the GUI to:
   - Capture the first and second images.
   - View calculated angle differences.
   - Export the results by clicking the "Export Data" button.

## Exported Data
Each export creates a timestamped folder named as follows:

```
YYYYMMDD_HH-MM-SS_angle-{angle_difference}_exp-{experiment_number}
```

### Folder Contents
- `photo1.png` and `photo2.png`: Full-resolution images.
- `photo1.jpg` and `photo2.jpg`: Segmentation masks.
- `Camera_pos_diagram.png`: Circle diagram visualization.
- `data.txt`: Detailed metrics, including:
  - Angle difference
  - Centroids
  - Dimensions
  - Subject area
  - Bounding boxes
  - Perimeters
  - Aspect ratios
  - Average brightness
  - Average colors

## Closing the Application
- Close the GUI window to end the webcam capture session and release resources properly.

## Notes
- Ensure the webcam is accessible and correctly configured.
- The application automatically repositions the window to the top center of the screen.
- The field of view (FOV) is assumed to be 60° for angle calculations.
