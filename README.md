
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
  4. **Result Exporting: **
     - Saves the matching figures to a timestamped output directory.'''

## Installation
Ensure all necessary Python packages are installed:

```bash
pip install -r requirements.txt
```

## Usage
1. **Update Configuration:**
   - Modify `sceneREID_config.yaml` to specify your image paths, model weights, thresholds, and other parameters.

2. **Run the Program:**

```bash
python FSU_sceneREID.py
```

3. **Review the Output:**
   - Results, including matching figures and object masks, will be saved in a uniquely named directory based on the current timestamp.

## Configuration Details (`sceneREID_config.yaml`)
- **Project details:**
  -`PROJECT_NAME`, `EXPERIMENT_NAME`: project specific identifiers so multiple projects can be maintained 
- **Image Paths:**
  - `IMG0_PTH`, `IMG1_PTH`: Paths to the two images to be processed.
- **Model Paths:**
  - `SAM_MODEL_ID`: Paths to the SAM3 model on Huggingface.
  - `VIT_MODEL_NAME`: Paths to the CLIP model on Huggingface (if another VIT model is used, the corresponding libraries will need to be imported into the python file.)
- **Processing Options:**
  - `VISUALIZE_FIG`: Option to display figures of original images, images with masks and matched objects with masks.
  - `RESIZE_HEIGHT`, `RESIZE_WIDTH`: Image dimensions for processing.
- **Algorithm Weights:**
  - `Y`: Weight to give Hu Moments to normalize
  -  `A`, `B`, `C`,: Weights to give to the algorithm based on the contribution of the component you want to emphasize
  -  total_simularity_score =  A*part_sim_score + B*lg_sim_score + C*hu_sim_score
