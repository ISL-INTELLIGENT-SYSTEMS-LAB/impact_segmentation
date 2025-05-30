import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox, filedialog, simpledialog
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import os
import datetime
import time
import numpy as np
import threading
import json
import cv2
from PIL import Image, ImageTk
from maskUtil import SegmentationApp
import torch
from segment_anything import sam_model_registry, SamPredictor

# Tell matplotlib to use the TkAgg backend for embedding plots in Tkinter
matplotlib.use("TkAgg")

# Load SAM model once and keep it in memory
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_model.to(device)
sam_predictor = SamPredictor(sam_model)

class CoordinatePlacer:
    def __init__(self, master):
        """
        Initializes the main application window and all required variables.
        Sets up both the GUI components and the webcam capture functionality.
        """
        self.master = master
        self.master.title("Object & Camera Placement")
        
        # Set the window to full screen
        try:
            self.master.state("zoomed")  # Windows
        except:
            self.master.attributes('-zoomed', True)  # Linux/macOS fallback

        # A Tkinter StringVar to track the current mode (“object” or “camera”)
        self.mode = tk.StringVar(value="object")

        # Lists to store object positions, camera positions, camera angles, etc.
        self.object_coords = []   # Each element is (x, z)
        self.camera_coords = []   # Each element is (x, z)
        self.camera_angles = []   # Each element is float angle
        self.lines_data = []      # Will store computed distances/angles for drawing lines
        self.captured_images = [] # Stores frames captured from the webcam
        self.object_names = []    # Match object_coords one-to-one
        self.ghost_artists = []  # Keep track of temporary preview artists

        # Default FOV before calibration
        self.camera_fov = 54  # degrees

        # Initialize webcam capture using OpenCV
        self.cap = cv2.VideoCapture(0)  # Use the default camera (0)

        # Set up the GUI: create widgets and plot area
        self.create_widgets()
        self.create_plot()

        # Run camera calibration after GUI is ready (so progress bar exists)
        self.master.after(100, lambda: threading.Thread(target=self.calibrate_camera_and_set_fov, daemon=True).start())

        # Continuously update the webcam feed in the GUI
        self.update_webcam()
        
        # Available ttkbootstrap themes and current index
        self.themes = self.master.style.theme_names()
        self.current_theme_index = self.themes.index(self.master.style.theme.name)
        
        
    def create_widgets(self):
        """
        Builds the main frames and control widgets of the GUI using ttkbootstrap styling.
        """
        # Main container frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame on the left for the matplotlib plot
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame on the right for controls and webcam
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Top portion of the right frame with controls
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        # Label for "Mode" and two radio buttons (Object/Camera)
        ttk.Label(control_frame, text="Mode:", font=("Segoe UI", 16, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(control_frame, text="Object", variable=self.mode, value="object").pack(side=tk.LEFT)
        ttk.Radiobutton(control_frame, text="Camera", variable=self.mode, value="camera").pack(side=tk.LEFT)

        # Angle entry box and label for degrees
        self.angle_entry = ttk.Entry(control_frame, width=5)
        self.angle_entry.insert(0, "0")
        self.angle_entry.pack(side=tk.LEFT)
        ttk.Label(control_frame, text="° Rotation").pack(side=tk.LEFT)

        # Themed action buttons with bootstyle
        self.capture_button = ttk.Button(control_frame, text="Capture", command=self.capture_image, state=tk.DISABLED, bootstyle="success")
        self.capture_button.pack(side=tk.LEFT, padx=5, ipadx=10, ipady=5)

        self.redo_button = ttk.Button(control_frame, text="Redo Last", command=self.redo_last_capture, bootstyle="secondary")
        self.redo_button.pack(side=tk.LEFT, padx=5, ipadx=10, ipady=5)

        self.export_button = ttk.Button(control_frame, text="Export", command=self.export_data, state=tk.DISABLED, bootstyle="warning")
        self.export_button.pack(side=tk.LEFT, padx=5, ipadx=10, ipady=5)

        self.import_button = ttk.Button(control_frame, text="Import", command=self.import_experiment, bootstyle="primary")
        self.import_button.pack(side=tk.LEFT, padx=5, ipadx=10, ipady=5)

        clear_button = ttk.Button(control_frame, text="Clear", command=self.clear_all, bootstyle="danger")
        clear_button.pack(side=tk.LEFT, padx=5, ipadx=10, ipady=5)

        reset_button = ttk.Button(control_frame, text="Reset Capture", command=self.reset_capture, bootstyle="danger")
        reset_button.pack(side=tk.LEFT, padx=5, ipadx=10, ipady=5)

        # Frame for indicator lights and webcam feed
        indicator_and_webcam = ttk.Frame(right_frame)
        indicator_and_webcam.pack(pady=10, fill=tk.BOTH, expand=True)

        # Label above webcam and row for indicators to the right
        top_indicator_row = ttk.Frame(indicator_and_webcam)
        top_indicator_row.pack(fill=tk.X)
        
        # Next shot information row
        next_shot_row = ttk.Frame(indicator_and_webcam)
        next_shot_row.pack(fill=tk.X, pady=(0, 10))
        # Label for next shot information
        self.next_shot_label = ttk.Label(next_shot_row, text="Next Shot: Place a camera on the plot to see its position and angle.", font=("Segoe UI", 16, "bold"))
        self.next_shot_label.pack(side=tk.LEFT, padx=(10, 5), pady=5)

        # "Shot Progress" label
        self.shot_label = ttk.Label(top_indicator_row, text="Shot Progress", font=("Segoe UI", 16, "bold"))
        self.shot_label.grid(row=0, column=0, padx=(10, 5), sticky="w")
        self.shot_label.grid_remove()

        # Frame that will hold the indicator circles, next to the label
        self.indicator_frame = ttk.Frame(top_indicator_row)
        self.indicator_frame.grid(row=0, column=1, sticky="w", padx=(0, 10))
        self.indicators = []

        # Progress bar for calibration (hidden by default)
        self.calibration_progress = ttk.Progressbar(
            indicator_and_webcam, orient="horizontal", length=200, mode="determinate")
        self.calibration_progress.pack(pady=5)
        self.calibration_progress.pack_forget()

        # Text label for calibration progress count (e.g., "Captured: 0 out of 30")
        self.calibration_label = ttk.Label(indicator_and_webcam, text="")
        self.calibration_label.pack()
        self.calibration_label.pack_forget()  # Hide until calibration begins

        # Skip calibration button (hidden by default)
        self.skip_calibration = False
        self.skip_button = ttk.Button(
            indicator_and_webcam,
            text="Continue without calibration",
            command=self.skip_calibration_early,
            bootstyle="secondary-outline"
        )
        self.skip_button.pack(pady=2, ipadx=10, ipady=5)
        self.skip_button.pack_forget()  # Hide until calibration starts

        # Label that will show the live webcam feed
        self.canvas_webcam = tk.Label(indicator_and_webcam, bg="black")
        self.canvas_webcam.config(width=640, height=480)
        self.canvas_webcam.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(10, 0))

        # Spacer to push theme button to bottom
        spacer = ttk.Frame(right_frame)
        spacer.pack(expand=True, fill=tk.BOTH)

        # Bottom toolbar for theme toggle button
        bottom_toolbar = ttk.Frame(right_frame)
        bottom_toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        toggle_theme_btn = ttk.Button(
            bottom_toolbar,
            text="Switch Themes",
            command=self.toggle_theme,
            bootstyle="secondary-outline"
        )
        toggle_theme_btn.pack(pady=5, padx=10)


    def create_plot(self):
        """
        Sets up the matplotlib figure and axes, embedding them into the Tkinter interface.
        """
        # Create a new figure and an axis for 2D plotting
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title("Click to place OBJECTS or CAMERAS (Z+ is North)")
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Z-axis")

        # Draw horizontal and vertical axes at 0 
        self.ax.axhline(0, color='blue', linewidth=1)
        self.ax.axvline(0, color='blue', linewidth=1)

        # Configure grid, aspect ratio, and axis limits/ticks
        self.ax.grid(True, color='blue', linestyle=':', linewidth=0.5)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_xticks(range(-10, 11, 1))
        self.ax.set_yticks(range(-10, 11, 1))

        # Plot a red dot at the origin to signify (0,0)
        self.ax.plot(0, 0, 'ro')

        # Embed the figure in a Tk Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Capture click events on the plot
        self.canvas.mpl_connect('button_press_event', self.onclick)
        
        # Capture hover events to show object/camera info
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        
    
    def on_hover(self, event):
        """
        Handles mouse movement to show ghost object or camera with FOV lines.
        If the cursor leaves the plot area or is invalid, the ghost disappears.
        """
        # If the cursor is outside the plot axes, remove ghost
        if not event.inaxes or event.xdata is None or event.ydata is None:
            self.clear_ghosts()
            self.canvas.draw_idle()
            return

        x, z = round(event.xdata), round(event.ydata)

        # Only allow within bounds
        if x < -10 or x > 10 or z < -10 or z > 10:
            self.clear_ghosts()
            self.canvas.draw_idle()
            return

        # Clear previous preview
        self.clear_ghosts()

        # Determine the current mode (object or camera)
        mode = self.mode.get()

        # Create a ghost object or camera based on the current mode
        if mode == "object":
            ghost = self.ax.plot(
                x, z,
                marker='o',
                markersize=10,
                markerfacecolor='deepskyblue',
                markeredgecolor='black',
                alpha=0.6,
                linestyle='None',
                zorder=10
            )
            self.ghost_artists.extend(ghost)

        # Draw lines to all cameras
        elif mode == "camera":
            try:
                angle = float(self.angle_entry.get())
            except ValueError:
                angle = 0

            # Draw the ghost camera with its FOV lines
            ghost = self.ax.plot(
                x, z,
                marker='^',
                markersize=10,
                markerfacecolor='lightgreen',
                markeredgecolor='black',
                alpha=0.6,
                linestyle='None',
                zorder=10
            )
            self.ghost_artists.extend(ghost)

            # Draw the FOV lines
            direction = math.radians(angle)
            spread = math.radians(self.camera_fov / 2)
            for offset in [-spread, spread]:
                dx = math.sin(direction + offset) * 3
                dz = math.cos(direction + offset) * 3
                line = self.ax.plot(
                    [x, x + dx], [z, z + dz],
                    color='green',
                    linewidth=1.5,
                    linestyle='--',
                    alpha=0.6,
                    zorder=9
                )
                self.ghost_artists.extend(line)

        # If we have a ghost artist, redraw the canvas
        self.canvas.draw_idle()
        
        
    def clear_ghosts(self):
        """
        Removes any ghost artists (temporary preview objects or cameras)
        """
        try:
            for artist in self.ghost_artists:
                artist.remove()
        finally:
            self.ghost_artists.clear()


    def update_webcam(self):
        """
        Grabs frames from the webcam and displays them in the GUI label. 
        This function re-calls itself after 10ms (for a continuous feed).
        """
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas_webcam.imgtk = imgtk
            self.canvas_webcam.configure(image=imgtk)

        # Save after callback ID so it can be cancelled
        self.after_id = self.master.after(10, self.update_webcam)


    def capture_image(self):
        """
        Captures a single frame from the webcam and marks the corresponding indicator 
        as 'green' once a photo is taken. If all cameras have corresponding images, 
        the 'Export' button becomes enabled.
        """
        ret, frame = self.cap.read()
        # Only capture if the read was successful and we haven't exceeded camera count
        if ret and len(self.captured_images) < len(self.camera_coords):
            self.captured_images.append(frame)
            # Turn the corresponding indicator green
            self.indicators[len(self.captured_images) - 1].itemconfig("light", fill="green")

        # If we have captured as many images as camera positions, disable capture and enable export
        if len(self.captured_images) >= len(self.camera_coords):
            self.capture_button.config(state=tk.DISABLED)
            self.export_button.config(state=tk.NORMAL)
            
        # Update the next shot label to reflect the current state
        self.update_next_shot_label()
          
            
    def reset_capture(self):
        """
        Resets the capture process without removing objects or cameras.
        Clears captured images and resets indicators.
        """
        self.captured_images.clear()
        for c in self.indicators:
            c.itemconfig("light", fill="white")
            
        # Reset the next shot label
        if len(self.camera_coords) == 0:
            self.shot_label.grid_remove()
            self.capture_button.config(state=tk.DISABLED)
            self.export_button.config(state=tk.DISABLED)
            return
        # If there are cameras, show the shot label and re-enable capture
        else:
            self.shot_label.grid()
            self.update_next_shot_label()
            self.capture_button.config(state=tk.NORMAL)
            self.export_button.config(state=tk.DISABLED)
        
        
    def redo_last_capture(self):
        """
        Removes the most recent captured image and resets the indicator light.
        Re-enables the capture button if needed.
        """
        if not self.captured_images:
            messagebox.showinfo("Nothing to redo", "No shots have been taken yet.")
            return

        # Remove last image
        self.captured_images.pop()

        # Reset the last indicator to white
        last_idx = len(self.captured_images)
        if last_idx < len(self.indicators):
            self.indicators[last_idx].itemconfig("light", fill="white")

        # Re-enable capture
        self.capture_button.config(state=tk.NORMAL)

        # Disable export until all images are captured again
        self.export_button.config(state=tk.DISABLED)
        
        # If there are cameras, update the next shot label
        self.update_next_shot_label()
            
        
    def skip_calibration_early(self):
        """
        User can click this to bypass the calibration step.
        """
        self.skip_calibration = True


    def onclick(self, event):
        """
        Handles mouse click events on the plot. Left-click to place an object/camera;
        right-click to remove an existing object/camera if clicked near it.
        """
        # Ignore out-of-bounds clicks
        if event.xdata is None or event.ydata is None:
            return

        # Round to grid
        x, z = round(event.xdata), round(event.ydata)
        if x < -10 or x > 10 or z < -10 or z > 10:
            return

        # Right-click (button=3) to remove object or camera
        if event.button == 3:
            for i, (ox, oz) in enumerate(self.object_coords):
                if math.hypot(x - ox, z - oz) < 1.0:
                    del self.object_coords[i]
                    del self.object_names[i]  # Remove corresponding name
                    self.redraw_plot()
                    return
            for i, (cx, cz) in enumerate(self.camera_coords):
                if math.hypot(x - cx, z - cz) < 1.0:
                    del self.camera_coords[i]
                    del self.camera_angles[i]
                    self.update_indicators()
                    self.redraw_plot()
                    return

        # Left-click to add object or camera
        if self.mode.get() == "object":
            name = self.prompt_for_object_name()  # Prompt for object name
            if not name:
                return
            self.object_coords.append((x, z))
            self.object_names.append(name)

        # If in camera mode, add camera with angle
        elif self.mode.get() == "camera":
            angle = float(self.angle_entry.get())
            self.camera_coords.append((x, z))
            self.camera_angles.append(angle)
            self.update_indicators()
            
        # Redraw the plot to show the new object or camera
        self.redraw_plot()


    def update_indicators(self):
        """
        Rebuilds the list of circle indicators on the right side to reflect how many cameras are placed.
        Each indicator corresponds to a camera and changes color once an image is captured.
        """
        for c in self.indicators:
            c.destroy()
        self.indicators.clear()

        # If there are cameras, show the shot label and enable capture/export buttons
        if len(self.camera_coords) > 0:
            self.shot_label.grid()
            self.capture_button.config(state=tk.NORMAL)
            self.export_button.config(state=tk.DISABLED)
        # If no cameras are placed, hide the shot label and disable buttons
        else:
            self.shot_label.grid_remove()
            self.capture_button.config(state=tk.DISABLED)
            self.export_button.config(state=tk.DISABLED)
            self.update_next_shot_label()
            return

        # Create indicators for each camera
        for i in range(len(self.camera_coords)):
            c = tk.Canvas(self.indicator_frame, width=20, height=20)
            color = "green" if i < len(self.captured_images) else "white"
            c.create_oval(2, 2, 18, 18, fill=color, outline="black", tags="light")
            row = i // 19
            col = i % 19
            c.grid(row=row, column=col, padx=2, pady=2)
            self.indicators.append(c)

        # Update the next shot label based on current camera state
        self.update_next_shot_label()
            
    
    def update_next_shot_label(self):
        """
        Updates the label with information about the next camera shot.
        """
        captured = len(self.captured_images)
        
        # If there are cameras and we haven't captured all of them, show the next shot info
        if captured < len(self.camera_coords):
            x, z = self.camera_coords[captured]
            angle = self.camera_angles[captured]
            self.next_shot_label.config(
                text=f"Next Shot: Camera at ({x}, {z}) with {angle}° rotation."
            )
        # If all cameras have been captured, show completion message
        elif self.camera_coords:
            self.next_shot_label.config(
                text="All shots captured. You may now export your data."
            )
        # If no cameras are placed, show default message
        else:
            self.next_shot_label.config(
                text="Next Shot: Place a camera on the plot to see its position and angle."
            )


    def clear_all(self):
        """
        Resets all internal data structures and clears the plot and indicators.
        Disables capture/export until new cameras are placed.
        """
        # Clear all internal data structures
        self.object_coords.clear()
        self.camera_coords.clear()
        self.camera_angles.clear()
        self.lines_data.clear()
        self.captured_images.clear()
        self.object_names.clear()

        # Remove existing indicator canvases
        for c in self.indicators:
            c.destroy()
        self.indicators.clear()

        # Hide the “Shot Progress” label and disable capture/export
        self.shot_label.grid_remove()
        self.capture_button.config(state=tk.DISABLED)
        self.export_button.config(state=tk.DISABLED)

        # Reset mode and angle entry to defaults
        self.mode.set("object")
        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, "0")

        # Redraw the empty plot
        self.redraw_plot()
        
        # Update the next shot label to default state
        self.update_next_shot_label()


    def redraw_plot(self):
        """
        Clears the plot and redraws all objects, cameras, camera FOV lines, and any 
        connection lines from the lines_data.
        """
        self.ghost_artists.clear()  # Clear any ghost artifacts

        # Clear the plot and reset title/labels
        self.ax.clear()
        self.ax.set_title("Click to place OBJECTS or CAMERAS (Z+ is North)")
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Z-axis")
        self.ax.axhline(0, color='blue', linewidth=1)
        self.ax.axvline(0, color='blue', linewidth=1)
        self.ax.grid(True, color='blue', linestyle=':', linewidth=0.5)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_xticks(range(-10, 11, 1))
        self.ax.set_yticks(range(-10, 11, 1))
        self.ax.plot(0, 0, 'ro')  # Origin marker

        # Redraw all placed objects
        for idx, (x, z) in enumerate(self.object_coords):
            self.ax.plot(x, z, 'bo')
            self.ax.text(x + 0.3, z + 0.3, self.object_names[idx], color='blue')

        # Redraw all placed cameras + FOV
        fov_half = self.camera_fov / 2
        for idx, ((x, z), angle) in enumerate(zip(self.camera_coords, self.camera_angles)):
            self.ax.plot(x, z, marker='^', color='green')
            self.ax.text(x + 0.3, z + 0.3, f"Cam {idx+1} ({angle:.0f}°)", color='green')

            rad1 = math.radians(angle - fov_half)
            rad2 = math.radians(angle + fov_half)
            length = 3
            x1, z1 = x + length * math.sin(rad1), z + length * math.cos(rad1)
            x2, z2 = x + length * math.sin(rad2), z + length * math.cos(rad2)
            self.ax.plot([x, x1], [z, z1], 'green', linestyle=':')
            self.ax.plot([x, x2], [z, z2], 'green', linestyle=':')

        # Draw lines to visible objects based on computed distances/angles
        self.canvas.draw_idle()


    def compute_distances_and_angles(self):
        """
        Calculates which objects each camera can “see” (eg. within a 90° FOV: ±45°),
        measuring distance from camera to each object and storing the result in lines_data.
        """
        # Clear old lines data
        self.lines_data.clear()

        # If the camera has a 90° field of view (±45° from its main angle)
        fov_half = self.camera_fov / 2

        # Loop over each camera
        for cam_idx, ((cx, cz), rotation) in enumerate(zip(self.camera_coords, self.camera_angles)):
            visible_objects = {}

            # Check each object to see if it falls within the camera’s FOV
            for obj_idx, (ox, oz) in enumerate(self.object_coords):
                dx = ox - cx
                dz = oz - cz

                # Angle from camera to object 
                angle_to_obj = math.degrees(math.atan2(dx, dz))

                # Normalize the relative angle to [–180, +180]
                rel_angle = (angle_to_obj - rotation + 360) % 360
                if rel_angle > 180:
                    rel_angle -= 360

                # If within ±45°, consider the object “visible”
                if -fov_half <= rel_angle <= fov_half:
                    distance = math.hypot(dx, dz)
                    visible_objects[f"Object{obj_idx+1}"] = {
                        "xpos": ox,
                        "zpos": oz,
                        "distance": round(distance, 2),
                        "angle": round(angle_to_obj, 2)
                    }

            # Store results for this camera
            self.lines_data.append({
                f"Camera{cam_idx+1}": {
                    "xpos": cx,
                    "zpos": cz,
                    "rotation": rotation,
                    "objects": visible_objects
                }
            })

        # Redraw the plot to show lines from cameras to their visible objects
        self.redraw_plot()
    
    
    def prompt_for_object_name(self):
        # Prompt user for a new object name
        name = simpledialog.askstring("Object Name", "Enter a name for this object:")
        if not name:
            return None

        # Ensure uniqueness by appending numbers if needed
        original = name
        suffix = 0
        while name in self.object_names:
            name = f"{original}{suffix}"
            suffix += 1

        return name
        
        
    def export_data(self):
        """
        Exports all collected data to a user-selected folder:
        - Creates a timestamped subfolder for data.json (containing cameras and object info)
        - Saves each captured image from the webcam
        - Exports the final scene plot as scene_plot.png
        - Launches the segmentation tool using the preloaded SAM model
        """
        # Compute distances and angles before exporting
        self.compute_distances_and_angles()

        # Ask for folder location
        folder = filedialog.askdirectory()
        if not folder:
            return

        # Create export directory with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(folder, f"placement_data_{timestamp}")
        os.makedirs(export_path, exist_ok=True)

        # Export named object coordinates
        objects_dict = {}
        for idx, (x, z) in enumerate(self.object_coords):
            objects_dict[self.object_names[idx]] = {
                "xpos": x,
                "zpos": z
            }

        # Export camera data with object angles
        cameras_dict = {}
        fov_half = self.camera_fov / 2
        for cam_idx, ((cx, cz), rot) in enumerate(zip(self.camera_coords, self.camera_angles)):
            visible_objects = {}
            for obj_idx, (ox, oz) in enumerate(self.object_coords):
                dx = ox - cx
                dz = oz - cz
                angle_to_obj = math.degrees(math.atan2(dx, dz))
                rel_angle = (angle_to_obj - rot + 360) % 360
                if rel_angle > 180:
                    rel_angle -= 360
                if -fov_half <= rel_angle <= fov_half:  # Object is within camera FOV
                    obj_name = self.object_names[obj_idx]
                    visible_objects[obj_name] = {
                        "distance": round(math.hypot(dx, dz), 2),
                        "angle": round(angle_to_obj, 2),
                        "degree": round(angle_to_obj, 2)
                    }
            # Store camera data
            cameras_dict[f"Camera{cam_idx+1}"] = {
                "xpos": cx,
                "zpos": cz,
                "rotation": rot,
                "objects": visible_objects
            }

        # Create the export JSON structure
        export_json = {
            "Objects": objects_dict,
            "Cameras": cameras_dict
        }

        # Save data.json
        with open(os.path.join(export_path, "data.json"), 'w') as jf:
            json.dump(export_json, jf, indent=2)

        # Save images
        for idx, img in enumerate(self.captured_images):
            x, z = self.camera_coords[idx]
            degree = self.camera_angles[idx]
            filename = f"Camera_{x}_{z}_{int(degree)}.png"
            img_path = os.path.join(export_path, filename)
            cv2.imwrite(img_path, img)

        # Save the scene plot
        self.fig.savefig(os.path.join(export_path, "scene_plot.png"))
        messagebox.showinfo("Export Successful", f"Data and images exported to:\n{export_path}")

        # Launch the segmentation tool
        SegmentationApp(export_path, self.object_coords, sam_predictor)
        
        
    def on_closing(self):
        """
        Properly release the webcam and destroy the main window.
        """
        if self.after_id is not None:
            self.master.after_cancel(self.after_id)
        self.cap.release()
        self.master.destroy()
        exit()
        
        
    def toggle_theme(self):
        """
        Toggles between available ttkbootstrap themes.
        Cycles through the list of themes and applies the next one.
        """
        self.current_theme_index = (self.current_theme_index + 1) % len(self.themes)
        next_theme = self.themes[self.current_theme_index]
        self.master.style.theme_use(next_theme)
             
                
    def import_experiment(self):
        """
        Lets the user choose a JSON experiment file to load object and camera positions.
        """
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not file_path:
            return

        try:
            # Load the JSON data from the selected file
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Clear all existing data before importing
            self.clear_all()

            # Load named objects explicitly
            objects = data.get("Objects", {})
            for name, obj in sorted(objects.items()):  # Sort to ensure consistent ordering across sessions
                self.object_coords.append((obj["xpos"], obj["zpos"]))
                self.object_names.append(name)

            # Load cameras and their data
            cameras = data.get("Cameras", {})
            for key in cameras:
                cam = cameras[key]
                self.camera_coords.append((cam["xpos"], cam["zpos"]))
                self.camera_angles.append(cam.get("rotation", 0.0))

            # Update indicators and redraw the plot
            self.update_indicators()
            self.redraw_plot()

            # Update the next shot label
            messagebox.showinfo("Import Successful", f"Loaded experiment from:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Import Failed", f"Error reading file:\n{e}")
            
        
    def calibrate_camera_and_set_fov(self):
        """
        Runs camera calibration 3 times using a 9x6 checkerboard pattern (25mm squares),
        then averages the estimated horizontal FOV values for stability.
        User can skip if desired.
        """
        max_calibration_images = 30
        total_runs = 3
        self.skip_calibration = False

        # Initialize webcam capture
        self.master.after(0, lambda: messagebox.showinfo(
            "Camera Calibration",
            "Camera calibration will run 3 times to improve accuracy.\n\n"
            "Please use a printed 9x6 checkerboard pattern (25mm squares, 10x7 total squares).\n"
            "Move the board to different positions and angles for each frame.\n\n"
            "Click 'Continue without calibration' to skip.\n"
        ))

        # Initialize progress bar and label
        self.calibration_progress["maximum"] = max_calibration_images
        self.calibration_progress["value"] = 0
        self.calibration_progress.pack()
        self.calibration_label.config(text=f"Captured: 0 out of {max_calibration_images}")
        self.calibration_label.pack()
        self.skip_button.pack()
        self.master.update_idletasks()

        # Start webcam capture
        fov_results = []
        for run_index in range(total_runs):
            if self.skip_calibration:
                break
            
            # Reset webcam capture for each run
            checkerboard_dims = (8, 6)  # 9x7 squares gives 8x6 inner corners
            square_size = 25.0  # in mm
            criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

            # Open webcam
            objp = np.zeros((np.prod(checkerboard_dims), 3), np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
            objp *= square_size

            # Initialize lists to hold object points and image points
            objpoints = []
            imgpoints = []

            # Reset the webcam capture
            count = 0
            self.calibration_progress["value"] = 0
            self.calibration_label.config(text=f"Run {run_index + 1} – Captured: 0 out of {max_calibration_images}")
            self.master.update_idletasks()

            # Start webcam feed
            while count < max_calibration_images and not self.skip_calibration:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # Convert frame to grayscale for corner detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret_corners, corners = cv2.findChessboardCorners(
                    gray, checkerboard_dims,
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                )

                # If corners are found, refine them and add to points
                if ret_corners:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)

                    # Draw the corners on the frame
                    frame = cv2.drawChessboardCorners(frame, checkerboard_dims, corners2, ret_corners)

                    # Convert frame to RGB for display in Tkinter
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                    self.canvas_webcam.imgtk = img_tk
                    self.canvas_webcam.configure(image=img_tk)

                    # Update the webcam label with the captured frame
                    self.master.update_idletasks()

                    # Pause visually for 500ms
                    start = time.time()
                    while time.time() - start < 0.5:
                        if self.skip_calibration:
                            break
                        self.master.update()
                        time.sleep(0.01)

                    # Increment count and update progress bar
                    count += 1
                    self.calibration_progress["value"] = count
                    self.calibration_label.config(
                        text=f"Run {run_index + 1} – Captured: {count} out of {max_calibration_images}"
                    )
                    self.master.update_idletasks()

            # If the user skipped calibration, break out of the loop
            if self.skip_calibration:
                break

            # Run calibration
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)

            # Calculate RMS reprojection error
            fx = camera_matrix[0, 0]
            w = gray.shape[1]
            fov_x = 2 * math.degrees(math.atan(w / (2 * fx)))
            fov_x = round(fov_x, 2)
            fov_results.append(fov_x)

            print(f"Run {run_index+1}: RMS Reprojection Error: {ret:.4f} | Estimated Horizontal FOV: {fov_x}°")

            # Draw the calibration result on the webcam label
            self.master.after(0, lambda: messagebox.showinfo(
                f"Calibration Run {run_index + 1} Complete",
                f"RMS Reprojection Error: {ret:.4f}\nEstimated Horizontal FOV: {fov_x}°"
            ))

        # Clean up GUI elements
        self.calibration_progress.pack_forget()
        self.calibration_label.pack_forget()
        self.skip_button.pack_forget()

        # Release the webcam
        if self.skip_calibration or len(fov_results) < 1:
            self.master.after(0, lambda: messagebox.showwarning("Calibration Skipped", "Using default FOV of 54°."))
            self.camera_fov = 54
        # If we have valid FOV results, calculate the average
        else:
            avg_fov = round(sum(fov_results) / len(fov_results), 2)
            self.camera_fov = avg_fov
            self.master.after(0, lambda: messagebox.showinfo(
                "Calibration Complete",
                f"Completed {len(fov_results)} calibration runs.\n\n"
                f"Average Horizontal FOV: {avg_fov}°"
            ))


if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = CoordinatePlacer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()