import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def get_json_data(json_fname):
  with open(json_fname, "r") as rfile:
    data = json.load(rfile)
  print(data["mask_legend"])


def sharpen_img(img_name):
  # Load image and convert to grayscale
  img = cv2.imread(img_name)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Threshold and clean noise
  #_, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
  # Otsu's thresholding after Gaussian filtering
  blur = cv2.GaussianBlur(gray,(5,5),0)
  ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
  #plt.imshow(th3)
  #plt.show()

  # Find contours and sort by area
  contours, _ = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
  
  min_area = 5  # You can change this threshold
  final_image = np.zeros_like(gray)
  # Create a mask for each shape via inRange
  for i, cnt in enumerate(sorted_contours):
      # Create blank mask
      area = cv2.contourArea(cnt)

      if area >= min_area:
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
        
        # Isolate using inRange (works with binary mask here)
        isolated = cv2.inRange(mask, 255, 255)
        
        # Calculate mean color using the mask
        mean_color = round(cv2.mean(img, mask=isolated)[0])

        isolated[isolated == 255] = mean_color
        
        print(f'Component {i+1} | Area: {int(area)} | Mean BGR: {mean_color}')
        final_image = cv2.bitwise_or(final_image, isolated)

        # Optional: Save each isolated region
        #cv2.imwrite(f'component_{i+1}.png', isolated)
  new_img_name = img_name.replace("jpg", "tif")
  cv2.imwrite(new_img_name, final_image)
  #plt.imshow(final_image)
  #plt.show()

#sharpen_img(r"E:\IMPACT\data_collection\scene_integration\rename_files\placement_data_20250528_133059\20250528_133059_Camera_-1_-1_0.jpg")


rootdir = r"E:\IMPACT\data_collection\scene_integration\working_dir"

for adir in os.listdir(rootdir):
  dirpath = os.path.join(rootdir, adir)
  for afile in os.listdir(dirpath):
    if afile.endswith("jpg"):
      print(afile)
      
      img_name = os.path.join(dirpath, afile)
      sharpen_img(img_name)
      tif_name = img_name.replace("jpg", "tif")
      
      img = cv2.imread(tif_name)
      colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)

      print(colors)
    