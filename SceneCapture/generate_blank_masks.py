import os
import json
import cv2
import numpy as np
from tkinter import filedialog, messagebox, Tk

def generate_missing_camera_masks(export_path):
    json_path = os.path.join(export_path, "data.json")
    if not os.path.exists(json_path):
        print("data.json not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    objects = data.get("Objects", {})
    gray_map = data.get("mask_legend", {})
    cameras = data.get("Cameras", {})

    added = 0

    for cam_id, cam_info in cameras.items():
        xpos = cam_info.get("xpos")
        zpos = cam_info.get("zpos")
        rotation = cam_info.get("rotation")

        if xpos is None or zpos is None or rotation is None:
            print(f"Skipping {cam_id}: missing position or rotation.")
            continue

        # Assemble filename
        filename_base = f"Camera_{xpos}_{zpos}_{int(rotation)}"
        image_file = filename_base + ".png"
        mask_file = filename_base + ".jpg"

        image_path = os.path.join(export_path, image_file)
        mask_path = os.path.join(export_path, mask_file)

        if os.path.exists(mask_path):
            continue  # Mask already exists

        if not os.path.exists(image_path):
            print(f"Missing image for {cam_id}: {image_file}")
            continue

        # Read image to get size
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue

        h, w = image.shape[:2]
        blank_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(mask_path, blank_mask)

        # Update JSON
        cam_masks = cam_info.setdefault("masks", {})
        for obj_name in objects.keys():
            if obj_name not in cam_masks and obj_name in gray_map:
                cam_masks[obj_name] = gray_map[obj_name]

        print(f"✅ Created blank mask for {cam_id} -> {mask_file}")
        added += 1

    # Save updated JSON
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Finished. {added} blank masks were added.")

def main():
    root = Tk()
    root.withdraw()
    export_path = filedialog.askdirectory(title="Select Experiment Folder")
    if not export_path:
        messagebox.showwarning("Cancelled", "No folder selected.")
        return
    generate_missing_camera_masks(export_path)
    messagebox.showinfo("Done", "Blank masks generated where needed.")

if __name__ == "__main__":
    main()
