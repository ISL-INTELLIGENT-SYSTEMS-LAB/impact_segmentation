
# Project Overview

This project demonstrates a custom image processing pipeline that utilizes advanced models for feature matching, object detection, and mask generation. The configuration and execution flow are designed for flexibility and modularity, enabling users to customize various parameters for their specific use cases.

## Main Components

- **Configuration:**
  - All adjustable parameters are defined in the `sceneREID_config.yaml` file, including paths for images, model weights, thresholds, and display options.
  - Users can easily modify image paths, model configurations, and processing settings.

- **Processing Workflow:**
  1. **Feature Matching (LightGlue):**
     - Matches keypoints between two images using the LG algorithm.
  2. **Object Detection and mask generation(SAM3):**
     - Detects specified objects in the first image based on text prompts.
     - Generates masks for detected objects and refines them using matching keypoints.
  3. **Shape matching (Hu moments):**
     - finds matches that have similar shapes
  '''4. **Result Exporting:**
     - Saves the matching figures, object masks, and refined results to a timestamped output directory.'''

## Installation
Ensure all necessary Python packages are installed:

```bash
pip install opencv-python numpy matplotlib torch torchvision pyyaml shapely
```

## Usage
1. **Update Configuration:**
   - Modify `config.yaml` to specify your image paths, model weights, thresholds, and other parameters.

2. **Run the Program:**

```bash
python main_script.py
```

3. **Review the Output:**
   - Results, including matching figures and object masks, will be saved in a uniquely named directory based on the current timestamp.

## Configuration Details (`config.yaml`)
- **Image Paths:**
  - `IMG0_PTH`, `IMG1_PTH`: Paths to the two images to be processed.
- **Model Paths:**
  - `LOFTR_INDOOR_PATH`, `LOFTR_OUTDOOR_PATH`: Paths to LoFTR model weights.
  - `GDINO_SCRIPT`, `GDINO_WEIGHTS`: Paths to GroundingDINO configurations and weights.
  - `SAM_CHECKPOINT`, `SAM_MODEL_CFG`: Paths to the SAM2 model and its configuration.
- **Processing Options:**
  - `BOX_THRESHOLD`, `TEXT_THRESHOLD`: Confidence thresholds for object detection.
  - `DISPLAY_POINTS_INSIDE`: Option to display matching points within detected objects.
  - `RESIZE_HEIGHT`, `RESIZE_WIDTH`: Image dimensions for processing.

## Subfolder Documentation
Each subfolder in this repository contains its own `README.md` file, providing detailed guidance on usage and functionality for that specific component. Users are encouraged to review these README files for deeper insights and step-by-step instructions on how each module operates.

## Notes
- The program ensures flexibility by adapting processing based on whether the images are indoor or outdoor.
- Error handling is integrated to notify users when matches or detections may be insufficient.
- The modular design supports easy extension and integration with other image processing pipelines.
