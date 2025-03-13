import pyarrow.parquet as pq
import numpy as np
import cv2
import os
from tqdm import tqdm

def extract_segmentations(segmentation_file, output_dir="extracted_segmentations"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load segmentation data from parquet file
    segmentation_table = pq.read_table(segmentation_file)

    # Define column names for segmentation components
    seg_column = '[CameraSegmentationLabelComponent].panoptic_label'
    seg_timestamp_col = 'key.frame_timestamp_micros'
    seg_camera_col = 'key.camera_name'

    # Iterate through segmentation data and extract masks with progress bar
    print()
    for idx in tqdm(range(segmentation_table.num_rows), desc="Extracting Segmentations"):
        seg_data = segmentation_table[seg_column][idx].as_py()
        if seg_data:
            # Decode segmentation mask from byte data
            mask = cv2.imdecode(np.frombuffer(seg_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Format filename and save segmentation mask
                filename = f"segmask_{segmentation_table[seg_timestamp_col][idx].as_py()}_{segmentation_table[seg_camera_col][idx].as_py()}.png"
                mask_path = os.path.join(output_dir, filename)
                if not os.path.exists(mask_path):
                    cv2.imwrite(mask_path, mask)

    print(f"Segmentation extraction completed. Saved to {output_dir}")