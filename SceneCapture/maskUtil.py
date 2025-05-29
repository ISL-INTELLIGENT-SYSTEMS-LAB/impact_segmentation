import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox
import tkinter as tk
import os
import json
import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class SegmentationApp:
    def __init__(self, export_path, object_coords, predictor):
        # Set paths and object info
        self.export_path = export_path
        self.object_coords = object_coords
        self.predictor = predictor

        # Load object names and assign grayscale values
        self.object_names = self.load_object_names()
        self.global_gray_map = {name: 30 + idx * 15 for idx, name in enumerate(self.object_names)}

        # Load image paths to segment
        self.image_paths = sorted([
            os.path.join(export_path, f)
            for f in os.listdir(export_path)
            if f.lower().startswith("camera_") and f.endswith(".png")
        ])
        self.mask_data = {}
        self.current_idx = 0

        # Create the main window and set it to zoomed/fullscreen
        self.root = ttk.Toplevel()
        self.root.title("Object Segmentation")
        try:
            self.root.state("zoomed")  # For Windows
        except:
            self.root.attributes('-zoomed', True)  # For Linux/macOS

        # Create the main container frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left frame for map/plot
        self.left_frame = ttk.Frame(main_frame, width=800)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True)

        # Right frame for segmentation tools
        self.right_frame = ttk.Frame(main_frame, width=400)
        self.right_frame.pack_propagate(False)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Inner frame centered vertically within the right_frame
        self.right_inner = ttk.Frame(self.right_frame)
        self.right_inner.pack(expand=True, padx=(0, 160))  # Adds 80px of right-side padding

        # Progress display
        self.progress_frame = ttk.Frame(self.right_inner)
        self.progress_frame.pack(padx=10, pady=(20, 10))

        self.filename_label = ttk.Label(self.progress_frame, text="", font=("Segoe UI", 12, "bold"))
        self.filename_label.pack(side=tk.LEFT, padx=(0, 10))

        self.progress = ttk.Progressbar(self.progress_frame, length=300, mode='determinate')
        self.progress.pack(side=tk.LEFT)

        self.progress_text = ttk.Label(self.progress_frame, text="", font=("Segoe UI", 12))
        self.progress_text.pack(side=tk.LEFT, padx=(10, 0))

        # Larger canvas for image display
        self.canvas = tk.Canvas(self.right_inner, width=800, height=600, bg="black", highlightthickness=0)
        self.canvas.pack(padx=10, pady=(20, 10))
        self.canvas.bind("<Button-1>", self.on_image_click)

        # Status label
        self.status = ttk.Label(self.right_inner, text="Click an object to segment", bootstyle="info", font=("Segoe UI", 12))
        self.status.pack(pady=(0, 15))

        # Button container
        btn_frame = ttk.Frame(self.right_inner)
        btn_frame.pack(pady=(0, 20))

        # Enlarged buttons with more spacing
        btn_font = ("Segoe UI", 11, "bold")

        self.toggle_button = ttk.Button(btn_frame, text="Next Mask", command=self.toggle_mask, bootstyle="primary", width=12)
        self.toggle_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.show_all_button = ttk.Button(btn_frame, text="Show All Masks", command=self.toggle_all_masks, bootstyle="info", width=14)
        self.show_all_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.save_button = ttk.Button(btn_frame, text="Save Mask", command=self.save_mask,
                                    state=tk.DISABLED, bootstyle="success", width=12)
        self.save_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.next_image_button = ttk.Button(btn_frame, text="Next Image", command=self.next_image,
                                            state=tk.NORMAL, bootstyle="warning", width=14)
        self.next_image_button.pack(side=tk.LEFT, padx=10, pady=5)

        # Setup the matplotlib plot for object/camera scene
        self.setup_plot()

        # Load the first image to begin segmentation
        self.load_image()

        # Launch the GUI event loop
        self.root.mainloop()

    def load_object_names(self):
        # Read object names from data.json
        json_path = os.path.join(self.export_path, "data.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                return list(data.get("Objects", {}).keys())
        return []

    def setup_plot(self):
        # Create a matplotlib figure and add it to the left frame
        self.fig, self.ax = plt.subplots(figsize=(12, 14), dpi=100)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.left_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_scene(self):
        # Plot all object locations and the camera with orientation arrow
        self.ax.clear()
        self.ax.set_title("Object and Camera Positions", fontsize=12)
        self.ax.set_xlim(-10.5, 10.5)
        self.ax.set_ylim(-10.5, 10.5)
        self.ax.set_xticks(range(-10, 11))
        self.ax.set_yticks(range(-10, 11))
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Z")
        self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

        # Plot objects as blue dots
        for name, (x, z) in zip(self.object_names, self.object_coords):
            self.ax.scatter(x, z, c="blue", s=80)
            self.ax.text(x + 0.3, z + 0.3, f"{name}", fontsize=8, color='blue')

        # Draw camera arrow if current image has a valid name
        if self.current_idx < len(self.image_paths):
            filename = os.path.basename(self.image_paths[self.current_idx])
            try:
                x = int(filename.split('_')[1])
                z = int(filename.split('_')[2])
                rotation = int(filename.split('_')[3].split('.')[0])

                # Draw camera marker
                self.ax.plot(x, z, marker='^', color='green', markersize=10)

                # FOV calculation (54° total → 27° each side)
                fov_half = 27
                length = 3
                rad1 = np.radians(rotation - fov_half)
                rad2 = np.radians(rotation + fov_half)
                x1, z1 = x + length * np.sin(rad1), z + length * np.cos(rad1)
                x2, z2 = x + length * np.sin(rad2), z + length * np.cos(rad2)

                self.ax.plot([x, x1], [z, z1], color='green', linestyle=':')
                self.ax.plot([x, x2], [z, z2], color='green', linestyle=':')

            except Exception as e:
                print(f"Error parsing camera filename: {e}")

        self.ax.set_aspect('equal')
        self.canvas_plot.draw()

    def load_image(self):
        # Stop if no images remain
        if self.current_idx >= len(self.image_paths):
            messagebox.showinfo("Done", "All images processed.")
            return

        # Load and display image
        self.image = cv2.imread(self.image_paths[self.current_idx])
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.image_rgb)

        # Clear state for new image
        self.current_masks = []
        self.selected_mask_idx = 0
        self.last_click_coords = None

        # Show the image and update plot and progress
        self.display_image()
        self.update_progress_display()
        self.plot_scene()
        
    def on_image_click(self, event):
        if self.tk_img is None:
            return

        x_disp, y_disp = event.x, event.y
        h_orig, w_orig, _ = self.image_rgb.shape

        # Map click coordinates from 800x600 back to original
        x = int(x_disp * (w_orig / 800))
        y = int(y_disp * (h_orig / 600))

        if x >= w_orig or y >= h_orig or x < 0 or y < 0:
            return

        self.last_click_coords = (x, y)

        # Predict masks
        masks, _, _ = self.predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True
        )

        self.current_masks = [m.astype(np.uint8) for m in masks]
        self.selected_mask_idx = 0
        self.display_image(mask=masks[0])
        self.save_button.config(state=tk.NORMAL)

    def display_image(self, mask=None):
        display = self.image_rgb.copy()
        if mask is not None:
            color = np.array([255, 0, 0])
            display[mask > 0] = (0.4 * display[mask > 0] + 0.6 * color).astype(np.uint8)

        # Resize image to fit 800x600 canvas
        img = Image.fromarray(display)
        img = img.resize((800, 600), Image.Resampling.LANCZOS)  # High-quality resize

        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def toggle_mask(self):
        # Show next available mask
        if not self.current_masks:
            return
        self.selected_mask_idx = (self.selected_mask_idx + 1) % len(self.current_masks)
        self.display_image(mask=self.current_masks[self.selected_mask_idx])

    def toggle_all_masks(self):
        # Display all saved masks
        if hasattr(self, 'composite_mask') and self.composite_mask is not None:
            self.display_image(mask=self.composite_mask)

    def update_progress_display(self):
        # Update progress bar and filename label
        filename = os.path.basename(self.image_paths[self.current_idx])
        try:
            xy_pos = f'({filename.split("_")[1]}, {filename.split("_")[2]})'
            degree = filename.split("_")[3].split('.')[0]
            self.filename_label.config(text=f"Camera at {xy_pos} | {degree}°")
        except:
            self.filename_label.config(text=f"Object Segmentation - {filename}")

        total = len(self.image_paths)
        current = self.current_idx + 1
        self.progress['maximum'] = total
        self.progress['value'] = current
        self.progress_text.config(text=f"({current} / {total})")

    def save_mask(self):
        # Save the selected mask under the chosen label
        if self.current_masks is None or len(self.current_masks) == 0 or self.last_click_coords is None:
            return

        base_name = os.path.splitext(os.path.basename(self.image_paths[self.current_idx]))[0]
        cam_id = base_name.replace("camera_", "Camera")
        mask = self.current_masks[self.selected_mask_idx]

        selected_name = self.prompt_object_name()
        if not selected_name:
            return

        if not hasattr(self, 'composite_mask') or self.composite_mask is None:
            h, w = mask.shape
            self.composite_mask = np.zeros((h, w), dtype=np.uint8)

        gray_value = self.global_gray_map.get(selected_name, 255)
        self.composite_mask[mask > 0] = gray_value
        self.mask_data.setdefault(cam_id, {})[selected_name] = gray_value

        self.status.config(text=f"Marked {selected_name}")
        self.save_button.config(state=tk.DISABLED)
        self.next_image_button.config(state=tk.NORMAL)

    def prompt_object_name(self):
        # Popup combo box to choose object name
        name_win = ttk.Toplevel(self.root)
        name_win.title("Select Object")
        width, height = 300, 150
        name_win.geometry(f"{width}x{height}")
        x = (name_win.winfo_screenwidth() // 2) - (width // 2)
        y = (name_win.winfo_screenheight() // 2) - (height // 2)
        name_win.geometry(f"+{x}+{y}")

        ttk.Label(name_win, text="Assign mask to object:", bootstyle="light").pack(padx=10, pady=5)

        # Sort object names alphabetically (with coords)
        name_coord_pairs = sorted(zip(self.object_names, self.object_coords), key=lambda pair: pair[0].lower())
        self.labeled_object_names = [
            f"{name}    ({coord[0]}, {coord[1]})" for name, coord in name_coord_pairs
        ]

        combo = ttk.Combobox(name_win, values=self.labeled_object_names, state="readonly", width=30,
                            font=("Segoe UI", 10, "bold"))
        combo.pack(padx=10, pady=5)
        combo.current(0)

        selected = {'name': None}

        def confirm():
            selected_label = combo.get()
            selected['name'] = selected_label.split(" (")[0].strip()
            name_win.destroy()

        ttk.Button(name_win, text="OK", command=confirm, bootstyle="success").pack(pady=5)
        self.root.wait_window(name_win)
        return selected['name']

    def next_image(self):
        # Save current composite mask and load next image
        base_name = os.path.splitext(os.path.basename(self.image_paths[self.current_idx]))[0]
        mask_path = os.path.join(self.export_path, f"{base_name}.jpg")
        cam_id = base_name.replace("camera_", "Camera")

        if not hasattr(self, 'composite_mask') or self.composite_mask is None:
            h, w, _ = self.image.shape
            self.composite_mask = np.zeros((h, w), dtype=np.uint8)
            for name in self.object_names:
                gray_value = self.global_gray_map.get(name, 255)
                self.mask_data.setdefault(cam_id, {})[name] = gray_value
            print(f"Generated blank mask for {cam_id}")

        cv2.imwrite(mask_path, self.composite_mask)
        self.current_idx += 1
        self.composite_mask = None
        self.save_button.config(state=tk.DISABLED)

        if self.current_idx < len(self.image_paths):
            self.load_image()
        else:
            self.save_and_exit()

    def save_and_exit(self):
        # Save all updated JSON data to file
        json_path = os.path.join(self.export_path, "data.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        for cam_id, mask_dict in self.mask_data.items():
            if cam_id not in data.get("Cameras", {}):
                continue
            data["Cameras"][cam_id]["masks"] = mask_dict

        data["mask_legend"] = self.global_gray_map

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        self.root.destroy()
        messagebox.showinfo("Done", "All images processed and saved.")



if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import json
    import os
    import torch
    from segment_anything import sam_model_registry, SamPredictor

    def load_sam_predictor():
        checkpoint_path = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam_model.to(device)
        return SamPredictor(sam_model)

    def get_object_data(export_path):
        json_path = os.path.join(export_path, "data.json")
        if not os.path.exists(json_path):
            return [], []
        with open(json_path, 'r') as f:
            data = json.load(f)
        object_dict = data.get("Objects", {})
        names = list(object_dict.keys())
        coords = [(v["xpos"], v["zpos"]) for v in object_dict.values()]
        return names, coords

    root = tk.Tk()
    root.withdraw()

    export_path = filedialog.askdirectory(title="Select Experiment Folder")
    if not export_path:
        messagebox.showwarning("Cancelled", "No folder selected.")
    else:
        json_path = os.path.join(export_path, "data.json")
        if not os.path.exists(json_path):
            messagebox.showerror("Missing Data", "data.json not found in selected folder.")
        else:
            names, coords = get_object_data(export_path)
            predictor = load_sam_predictor()
            SegmentationApp(export_path, coords, predictor)
            root.destroy()  # <- Ensures program terminates cleanly