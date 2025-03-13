"""
    Taylor J. Brown | March 3rd 2025 | ISL - Intelligent Systems Lab
    
    Webcam Capture & Analysis Application
    
    Run this command first before running the application:
    pip install -r requirements.txt

    This application provides a graphical interface to capture images from a webcam,
    perform background removal to segment the main subject, and compute various image metrics.
    The key features include:

    - Capturing two images from the live webcam feed.
    - Automatic segmentation using the 'rembg' library to isolate the subject.
    - Calculation of multiple metrics for each captured image, including:
        - Subject centroid,
        - Contour area,
        - Bounding box,
        - Contour perimeter,
        - Aspect ratio,
        - Average brightness,
        - Average color.
    - Computation of the angle difference between the two images based on the subject's horizontal offset.
    - Generation of a circle diagram that visualizes:
        - The subject's position (blue dot),
        - Camera positions (red dots),
        - Red lines indicating the calculated angles,
        - A legend explaining the diagram elements.
    - Exporting data into a timestamped folder that includes:
        - Full-resolution captured images,
        - Segmentation masks,
        - The circle diagram saved as "Camera_pos_diagram.png",
        - A detailed data.txt file containing all computed metrics.
    - A scaled user interface that displays a resized live video feed and captured images,
    with the application window automatically positioned at the top center of the screen.

    Requirements:
        - numpy
        - opencv-python
        - Pillow
        - rembg
        - onnxruntime
"""

import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import datetime
import os
from PIL import Image, ImageTk, ImageDraw
import math
from rembg import remove

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Webcam Capture and Analysis")

        # Initialize webcam
        self.cap = cv2.VideoCapture(1)

        # Variables to store captures, masks, centroids, and additional image data
        self.photo1 = None  # First captured image (RGB)
        self.photo2 = None  # Second captured image (RGB)
        self.mask1 = None
        self.mask2 = None
        self.centroid1 = None
        self.centroid2 = None
        self.area1 = None   # Largest contour area for photo1
        self.area2 = None   # Largest contour area for photo2
        self.bbox1 = None   # Bounding box (x, y, w, h) for photo1
        self.bbox2 = None   # Bounding box for photo2
        self.perimeter1 = None  # Contour perimeter for photo1
        self.perimeter2 = None  # Contour perimeter for photo2
        self.aspect_ratio1 = None  # Aspect ratio (w/h) for photo1
        self.aspect_ratio2 = None  # Aspect ratio for photo2
        self.brightness1 = None  # Average brightness for photo1 (subject area)
        self.brightness2 = None  # Average brightness for photo2
        self.avg_color1 = None   # Average color (R, G, B) for photo1
        self.avg_color2 = None   # Average color for photo2
        self.angle_difference = None

        # Assumed horizontal field of view (degrees)
        self.fov = 60

        # Capture state: 0 = none, 1 = first captured, 2 = both captured
        self.current_capture = 0

        # GUI Elements

        # Live webcam feed
        self.video_label = tk.Label(master)
        self.video_label.pack()

        # Capture button
        self.capture_button = tk.Button(master, text="Take First Picture", command=self.capture_photo)
        self.capture_button.pack()
        
        # Angle difference label
        self.angle_label = tk.Label(master, text="Angle Difference: N/A")
        self.angle_label.pack(pady=5)
        
        # Export button
        self.export_button = tk.Button(master, text="Export Data", command=self.export_data)
        self.export_button.pack(pady=5)

        # Frame for captured images
        self.captured_frame = tk.Frame(master)
        self.captured_frame.pack()

        # Placeholder image (light gray)
        self.placeholder_imgtk = self.create_placeholder_image(200, 150)

        # Labels for captured images
        self.photo1_label = tk.Label(self.captured_frame, image=self.placeholder_imgtk)
        self.photo1_label.grid(row=0, column=0, padx=5, pady=5)
        self.photo2_label = tk.Label(self.captured_frame, image=self.placeholder_imgtk)
        self.photo2_label.grid(row=0, column=1, padx=5, pady=5)

        # Start updating the webcam feed
        self.update_video()

    def create_placeholder_image(self, width, height):
        """Creates a blank (light gray) placeholder image."""
        placeholder = np.full((height, width, 3), 200, dtype=np.uint8)
        pil_placeholder = Image.fromarray(placeholder)
        return ImageTk.PhotoImage(image=pil_placeholder)

    def update_video(self):
        """Continuously grabs frames from the webcam and updates the display."""
        ret, frame_bgr = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.current_frame = frame_rgb.copy()
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.master.after(10, self.update_video)

    def capture_photo(self):
        """Handles the capture sequence. On the first capture, save photo1; on the second, save photo2 and compute angles."""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            return

        if self.current_capture == 0:
            self.photo1 = self.current_frame.copy()  # Already in RGB
            (self.mask1, self.centroid1, self.area1, self.bbox1, 
             self.perimeter1, self.aspect_ratio1, self.brightness1, self.avg_color1) = self.segment_person(self.photo1)
            img_pil = Image.fromarray(self.photo1)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            self.photo1_label.imgtk = imgtk
            self.photo1_label.configure(image=imgtk)
            self.current_capture = 1
            self.capture_button.config(text="Take Second Picture")
        elif self.current_capture == 1:
            self.photo2 = self.current_frame.copy()
            (self.mask2, self.centroid2, self.area2, self.bbox2, 
             self.perimeter2, self.aspect_ratio2, self.brightness2, self.avg_color2) = self.segment_person(self.photo2)
            img_pil = Image.fromarray(self.photo2)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            self.photo2_label.imgtk = imgtk
            self.photo2_label.configure(image=imgtk)
            self.current_capture = 2
            self.calculate_angle_difference()

    def segment_person(self, image_rgb):
        """
        Uses rembg to remove the background from the image.
        Returns:
          - mask: Binary mask (person=255, background=0)
          - centroid: (x, y) of the largest detected contour (subject)
          - area: Area of the largest contour
          - bbox: Bounding box (x, y, w, h) of the largest contour
          - perimeter: Perimeter of the largest contour
          - aspect_ratio: bbox width divided by height
          - subject_brightness: Average brightness (grayscale mean) of the subject region
          - avg_color: Average color (R, G, B tuple) of the subject region
        """
        pil_img = Image.fromarray(image_rgb)
        out_pil = remove(pil_img)
        out_np = np.array(out_pil)  # (H, W, 4) RGBA

        # Use alpha channel as mask
        alpha = out_np[:, :, 3]
        mask = np.where(alpha > 128, 255, 0).astype(np.uint8)

        # Compute average brightness and average color within the subject region
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        subject_brightness = cv2.mean(gray, mask=mask)[0]
        avg_color = cv2.mean(image_rgb, mask=mask)[:3]  # (R, G, B)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape
        centroid = (w // 2, h // 2)
        area = 0.0
        bbox = (0, 0, 0, 0)
        perimeter = 0.0
        aspect_ratio = 0.0

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            bbox = cv2.boundingRect(largest_contour)  # (x, y, w, h)
            perimeter = cv2.arcLength(largest_contour, True)
            aspect_ratio = bbox[2] / bbox[3] if bbox[3] != 0 else 0
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroid = (cX, cY)
        return mask, centroid, area, bbox, perimeter, aspect_ratio, subject_brightness, avg_color

    def calculate_angle_difference(self):
        """
        Calculates approximate angles for both photos based on the horizontal offset of the subject's centroid.
        Updates the angle difference label and generates the circle diagram.
        """
        width = self.photo1.shape[1]
        center_x = width / 2.0

        # Calculate angle using the horizontal offset (assuming fixed FOV)
        angle1 = ((self.centroid1[0] - center_x) / center_x) * (self.fov / 2.0)
        angle2 = ((self.centroid2[0] - center_x) / center_x) * (self.fov / 2.0)

        self.angle_difference = abs(angle2 - angle1)
        self.angle_label.config(text=f"Angle Difference: {self.angle_difference:.2f}")

    def generate_angle_circle_image(self, angle1, angle2, width=300, height=300):
        """
        Draws a circle diagram with:
          - A blue dot at the center (subject)
          - Two red points representing the two camera angles
          - Lines from the center to each red point
          - A legend at the bottom of the diagram
        The image is expanded vertically to accommodate the legend.
        """
        angle_height = 10
        legend_height = 30
        total_height = height + legend_height + angle_height
        img = Image.new('RGB', (width, total_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw the circle in the top portion
        cx, cy = width // 2, height // 2
        radius = min(cx, height // 2) - 10

        # Draw outer circle
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     outline='black', width=2)

        # Draw center point (subject) as a blue dot
        center_dot_r = 5
        draw.ellipse([cx - center_dot_r, cy - center_dot_r, cx + center_dot_r, cy + center_dot_r],
                     fill='blue')

        # Convert angles to radians and compute positions for red dots
        rad1 = math.radians(angle1)
        rad2 = math.radians(angle2)
        x1 = cx + radius * math.cos(rad1)
        y1 = cy + radius * math.sin(rad1)
        x2 = cx + radius * math.cos(rad2)
        y2 = cy + radius * math.sin(rad2)

        dot_r = 5
        # Draw red dots
        draw.ellipse([x1 - dot_r, y1 - dot_r, x1 + dot_r, y1 + dot_r], fill='red')
        draw.ellipse([x2 - dot_r, y2 - dot_r, x2 + dot_r, y2 + dot_r], fill='red')

        # Draw lines from the center to the red dots
        draw.line((cx, cy, x1, y1), fill='red', width=2)
        draw.line((cx, cy, x2, y2), fill='red', width=2)
        
        # Draw the angle text at the top
        angle_text = f"Angle Difference: {self.angle_difference:.2f} degrees"
        text_x = 5
        text_y = height + (angle_height - 10) // 2
        draw.text((text_x, text_y), angle_text, fill="black")

        # Draw legend in the bottom margin
        legend_text = "Blue = Subject    Red = Camera Position    Red Line = Angle"
        # Using default font; you can also specify a truetype font if needed
        text_x = 5
        text_y = height + (legend_height - 10) // 2
        draw.text((text_x, text_y), legend_text, fill="black")
        return img

    def export_data(self):
        """
        Exports the captured photos, masks, and the circle diagram (saved as "Camera_pos_diagram.png"),
        along with a data.txt file that now contains:
          - Angle difference, centroids, photo dimensions, subject area,
          - Bounding box, perimeter, aspect ratio,
          - Average brightness and average color of the subject region.
        The folder is named as:
          YYYYMMDD_HH-MM-SS_angle-{angle_difference}_exp-{experiment_number}
        After export, the GUI resets.
        """
        if self.current_capture < 2:
            return

        folder_selected = filedialog.askdirectory()
        if not folder_selected:
            return

        # Determine experiment number by counting folders starting with today's date
        today_prefix = datetime.datetime.now().strftime("%Y%m%d")
        experiment_number = len([name for name in os.listdir(folder_selected)
                                 if name.startswith(today_prefix)])

        # Create timestamped folder using "exp"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
        folder_name = f"{timestamp}_angle-{self.angle_difference:.2f}_exp-{experiment_number}"
        full_path = os.path.join(folder_selected, folder_name)
        os.makedirs(full_path, exist_ok=True)

        # Save photos (convert from RGB to BGR)
        photo1_path = os.path.join(full_path, "photo1.png")
        photo2_path = os.path.join(full_path, "photo2.png")
        cv2.imwrite(photo1_path, cv2.cvtColor(self.photo1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(photo2_path, cv2.cvtColor(self.photo2, cv2.COLOR_RGB2BGR))

        # Save masks as JPG
        mask1_path = os.path.join(full_path, "photo1.jpg")
        mask2_path = os.path.join(full_path, "photo2.jpg")
        cv2.imwrite(mask1_path, self.mask1)
        cv2.imwrite(mask2_path, self.mask2)

        # Save circle diagram as "Camera_pos_diagram.png"
        width = self.photo1.shape[1]
        center_x = width / 2.0
        angle1 = ((self.centroid1[0] - center_x) / center_x) * (self.fov / 2.0)
        angle2 = ((self.centroid2[0] - center_x) / center_x) * (self.fov / 2.0)
        circle_img = self.generate_angle_circle_image(angle1, angle2)
        circle_img_path = os.path.join(full_path, "Camera_pos_diagram.png")
        circle_img.save(circle_img_path)

        # Save analysis data to data.txt (without degree symbols)
        data_path = os.path.join(full_path, "data.txt")
        with open(data_path, "w") as f:
            f.write(f"Angle Difference: {self.angle_difference:.2f}\n")
            f.write(f"Photo1 Centroid: {self.centroid1}\n")
            f.write(f"Photo2 Centroid: {self.centroid2}\n")
            f.write(f"Photo1 Dimensions: {self.photo1.shape[1]}x{self.photo1.shape[0]}\n")
            f.write(f"Photo2 Dimensions: {self.photo2.shape[1]}x{self.photo2.shape[0]}\n")
            f.write(f"Photo1 Subject Area: {self.area1:.2f}\n")
            f.write(f"Photo2 Subject Area: {self.area2:.2f}\n")
            f.write(f"Photo1 Bounding Box: {self.bbox1}\n")
            f.write(f"Photo2 Bounding Box: {self.bbox2}\n")
            f.write(f"Photo1 Perimeter: {self.perimeter1:.2f}\n")
            f.write(f"Photo2 Perimeter: {self.perimeter2:.2f}\n")
            f.write(f"Photo1 Aspect Ratio: {self.aspect_ratio1:.2f}\n")
            f.write(f"Photo2 Aspect Ratio: {self.aspect_ratio2:.2f}\n")
            f.write(f"Photo1 Average Brightness: {self.brightness1:.2f}\n")
            f.write(f"Photo2 Average Brightness: {self.brightness2:.2f}\n")
            f.write(f"Photo1 Average Color: {self.avg_color1}\n")
            f.write(f"Photo2 Average Color: {self.avg_color2}\n")

        print(f"Data exported successfully to {full_path.split('\\')[-1]}")
        self.reset_gui()

    def reset_gui(self):
        """Resets all stored data and GUI elements to their initial state."""
        self.photo1 = None
        self.photo2 = None
        self.mask1 = None
        self.mask2 = None
        self.centroid1 = None
        self.centroid2 = None
        self.area1 = None
        self.area2 = None
        self.bbox1 = None
        self.bbox2 = None
        self.perimeter1 = None
        self.perimeter2 = None
        self.aspect_ratio1 = None
        self.aspect_ratio2 = None
        self.brightness1 = None
        self.brightness2 = None
        self.avg_color1 = None
        self.avg_color2 = None
        self.angle_difference = None
        self.current_capture = 0

        self.photo1_label.configure(image=self.placeholder_imgtk)
        self.photo2_label.configure(image=self.placeholder_imgtk)
        self.photo1_label.imgtk = self.placeholder_imgtk
        self.photo2_label.imgtk = self.placeholder_imgtk

        self.angle_label.config(text="Angle Difference: N/A")
        self.capture_button.config(text="Take First Picture")

    def on_closing(self):
        """Releases resources on application close."""
        self.cap.release()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)

    # Function to reposition the window at the top center of the screen.
    def reposition_window(event=None):
        screen_width = root.winfo_screenwidth()
        window_width = root.winfo_width()
        new_x = (screen_width - window_width) // 2
        root.geometry(f"+{new_x}+0")

    # Bind the configure event so that when the window is updated it repositions.
    root.bind("<Configure>", reposition_window)

    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
