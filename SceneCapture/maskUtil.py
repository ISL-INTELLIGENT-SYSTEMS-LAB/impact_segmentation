# maskUtil.py
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox
import tkinter as tk
import os
import json
import cv2
import numpy as np
from PIL import Image, ImageTk

class SegmentationApp:
    def __init__(self, export_path, object_coords, predictor):
        # Paths and object info
        self.export_path = export_path
        self.object_coords = object_coords

        # Predictor passed from the main script (already initialized with SAM model)
        self.predictor = predictor

        # Assign fixed object names and grayscale values for the entire experiment
        self.object_names = self.load_object_names()
        self.global_gray_map = {
            name: 30 + idx * 15 for idx, name in enumerate(self.object_names)
        }

        # Find all camera images to process
        self.image_paths = sorted([
            os.path.join(export_path, f)
            for f in os.listdir(export_path) if f.startswith("camera_") and f.endswith(".png")
        ])
        self.mask_data = {}  # Stores per-camera mask label info
        self.current_idx = 0  # Index of the current image being processed

        # Set up GUI window using ttkbootstrap
        self.root = ttk.Toplevel()
        self.root.title("Object Segmentation")
        self.root.geometry("700x600")

        # Canvas for displaying images
        self.canvas = tk.Canvas(self.root, width=640, height=480, bg="black")
        self.canvas.pack(padx=10, pady=(10, 5))
        self.canvas.bind("<Button-1>", self.on_click)

        # Status label (dark themed)
        self.status = ttk.Label(self.root, text="Click an object to segment", bootstyle="info")
        self.status.pack(pady=(0, 10))

        # Button controls in a frame
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=(0, 10))

        self.toggle_button = ttk.Button(btn_frame, text="Next Mask", command=self.toggle_mask, bootstyle="primary")
        self.toggle_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(btn_frame, text="Save Mask", command=self.save_mask,
                                    state=tk.DISABLED, bootstyle="success")
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.next_image_button = ttk.Button(btn_frame, text="Next Image", command=self.next_image,
                                            state=tk.DISABLED, bootstyle="warning")
        self.next_image_button.pack(side=tk.LEFT, padx=5)

        # Load the first image and center the window
        self.load_image()
        self.center_window()
        self.root.mainloop()
        
    def center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def load_object_names(self):
        # Load object names from data.json if available
        json_path = os.path.join(self.export_path, "data.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                return list(data.get("Objects", {}).keys())
        return []

    def load_image(self):
        # End if all images have been processed
        if self.current_idx >= len(self.image_paths):
            messagebox.showinfo("Done", "All images processed.")
            return

        # Read and convert image to RGB
        self.image = cv2.imread(self.image_paths[self.current_idx])
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Set current image in SAM predictor
        self.predictor.set_image(self.image_rgb)

        # Reset masks and click info for this image
        self.current_masks = []
        self.selected_mask_idx = 0
        self.last_click_coords = None

        # Show the image
        self.display_image()

    def display_image(self, mask=None):
        # Display base image or image with overlaid mask
        display = self.image_rgb.copy()
        if mask is not None:
            color = np.array([255, 0, 0])  # Red overlay
            display[mask > 0] = (0.4 * display[mask > 0] + 0.6 * color).astype(np.uint8)

        img = Image.fromarray(display)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def on_click(self, event):
        # Get clicked pixel location
        x, y = event.x, event.y
        h, w, _ = self.image_rgb.shape
        if x >= w or y >= h:
            return

        # Use click as a point prompt for SAM
        self.last_click_coords = np.array([[x, y]])
        input_labels = np.array([1])  # 1 = foreground point

        # Predict multiple possible masks
        masks, scores, _ = self.predictor.predict(
            point_coords=self.last_click_coords,
            point_labels=input_labels,
            multimask_output=True
        )

        self.current_masks = [m.astype(np.uint8) for m in masks]
        self.selected_mask_idx = 0  # Start with the first candidate mask
        self.display_image(mask=self.current_masks[self.selected_mask_idx])

        # Enable save
        self.save_button.config(state=tk.NORMAL)
        self.next_image_button.config(state=tk.DISABLED)

    def toggle_mask(self):
        # Cycle through candidate masks
        if not self.current_masks:
            return
        self.selected_mask_idx = (self.selected_mask_idx + 1) % len(self.current_masks)
        self.display_image(mask=self.current_masks[self.selected_mask_idx])

    def save_mask(self):
        # Skip if no valid mask or click
        if not self.current_masks or self.last_click_coords is None:
            return

        # Get base file name and camera ID
        base_name = os.path.splitext(os.path.basename(self.image_paths[self.current_idx]))[0]
        cam_id = base_name.replace("camera_", "Camera")
        mask = self.current_masks[self.selected_mask_idx]

        # Prompt user to assign object name via combo box
        selected_name = self.prompt_object_name()
        if not selected_name:
            return

        # Initialize composite grayscale mask if it doesn't exist
        if not hasattr(self, 'composite_mask') or self.composite_mask is None:
            h, w = mask.shape
            self.composite_mask = np.zeros((h, w), dtype=np.uint8)

        # Get the global grayscale value for this object
        gray_value = self.global_gray_map.get(selected_name, 255)  # Fallback to 255 if unknown

        # Apply grayscale value to composite mask where this object is detected
        self.composite_mask[mask > 0] = gray_value

        # Add to per-camera mask JSON
        self.mask_data.setdefault(cam_id, {})[selected_name] = gray_value

        # Update status and UI state
        self.status.config(text=f"Marked {selected_name} with value {gray_value}")
        self.save_button.config(state=tk.DISABLED)
        self.next_image_button.config(state=tk.NORMAL)

    def prompt_object_name(self):
        # GUI popup using ttkbootstrap to select known object names
        name_win = ttk.Toplevel(self.root)
        name_win.title("Select Object")

        width, height = 300, 120
        name_win.geometry(f"{width}x{height}")
        x = (name_win.winfo_screenwidth() // 2) - (width // 2)
        y = (name_win.winfo_screenheight() // 2) - (height // 2)
        name_win.geometry(f"+{x}+{y}")

        ttk.Label(name_win, text="Assign mask to object:", bootstyle="light").pack(padx=10, pady=5)
        combo = ttk.Combobox(name_win, values=self.object_names, state="readonly", width=25)
        combo.pack(padx=10, pady=5)
        combo.current(0)

        selected = {'name': None}
        def confirm():
            selected['name'] = combo.get()
            name_win.destroy()

        ttk.Button(name_win, text="OK", command=confirm, bootstyle="success").pack(pady=5)
        self.root.wait_window(name_win)
        return selected['name']

    def next_image(self):
        # Save composite mask as a .jpg image
        base_name = os.path.splitext(os.path.basename(self.image_paths[self.current_idx]))[0]
        mask_path = os.path.join(self.export_path, f"{base_name}.jpg")
        if hasattr(self, 'composite_mask') and self.composite_mask is not None:
            cv2.imwrite(mask_path, self.composite_mask)

        # Advance to the next image and reset states
        self.current_idx += 1
        self.composite_mask = None
        self.save_button.config(state=tk.DISABLED)
        self.next_image_button.config(state=tk.DISABLED)

        # If more images, load next; otherwise, finish
        if self.current_idx < len(self.image_paths):
            self.load_image()
        else:
            self.save_and_exit()

    def closest_object_name(self, mask):
        # Get the average (x, z) grid position of the mask
        y, x = np.argwhere(mask).mean(axis=0)
        h, w = mask.shape
        xg = int((x / w) * 20 - 10)
        zg = int((y / h) * 20 - 10)

        # Find closest object in grid to assign a name
        min_dist = float("inf")
        closest_name = "Unknown"
        for idx, (ox, oz) in enumerate(self.object_coords):
            dist = (ox - xg) ** 2 + (oz - zg) ** 2
            if dist < min_dist:
                min_dist = dist
                closest_name = f"Object{idx+1}"
        return closest_name

    def save_and_exit(self):
        # Load existing JSON or initialize new structure
        json_path = os.path.join(self.export_path, "data.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        # Append camera-wise mask info
        for cam_id, mask_dict in self.mask_data.items():
            if cam_id not in data.get("Cameras", {}):
                continue
            data["Cameras"][cam_id]["masks"] = mask_dict

        # Save the grayscale legend for global decoding
        data["mask_legend"] = self.global_gray_map

        # Save updated JSON
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Close the GUI
        self.root.destroy()
        messagebox.showinfo("Done", "All images processed and saved.")
