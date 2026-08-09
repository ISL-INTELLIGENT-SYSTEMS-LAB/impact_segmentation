from unrealcv import Client
import imageio
import base64
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import json 
import os
from datetime import datetime
import sys
import math

def parse_color(s):
    # Example input: "(R=183,G=172,B=200,A=22)"
    s = s.strip("()")  # remove parentheses
    parts = s.split(",")
    vals = {}
    for p in parts:
        key, val = p.split("=")
        vals[key] = int(val)
    return vals["R"], vals["G"], vals["B"], vals["A"]
    
def capture_360(client, target_obj, actor_list, radius=300, height=100, steps=36, outdir="capture"):
    """
    Capture RGB + segmentation masks + bounding boxes
    in a 360-degree circle around a target object.
    """
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs(outdir, exist_ok=True)

    # Get target object location
    loc = client.request(f"vget /object/{target_obj}/location")
    tx, ty, tz = map(float, loc.split())

    # Get list of all UnrealCV objects
    #all_objects = client.request("vget /objects").split()

    for i in range(steps):
        angle = (360.0 / steps) * i
        rad = math.radians(angle)

        # Camera position on circle
        cx = tx + radius * math.cos(rad)
        cy = ty + radius * math.sin(rad)
        cz = tz + height

        # Move camera
        client.request(f"vset /camera/0/location {cx} {cy} {cz}")

        # Point camera at target
        client.request(f"vset /camera/0/rotation {0} {angle + 180} {0}")

        rgb_path = f"{outdir}/{now}_{target_obj}_rgb_{int(angle):03d}.png"
        mask_path = rgb_path.replace("rgb", "mask")

        # Capture RGB
        rgb_bytes = client.request('vget /camera/0/lit')
        print(rgb_bytes)
        with open(rgb_path, 'wb') as f:
          f.write(rgb_bytes)

        # Capture mask
        mask_bytes = client.request('vget /camera/0/object_mask png')
        with open(mask_path, 'wb') as f:
          f.write(mask_bytes)
          
        print(f"Captured angle {i} ({angle} degrees)")





client = Client(('localhost', 9000))
client.connect()

colormapfile_path = r"C:\Users\casab\Documents\Unreal Projects\military_base_3maps\Saved\UnrealCV\colormap_output.json"

actors = ['BP_Military_Pickup_C_1', 'BP_Trailer_C_2', 'BP_Military_Car_C_6', 'BP_Military_Car_C_7', 'BP_Military_Car_C_0', 
'BP_Military_Pickup_C_0', 'BP_Military_Pickup_C_2', 'BP_Military_Pickup_C_5', 'BP_SAM_Launcher_C_3', 'BP_Military_Pickup_C_3', 
'BP_Trailer_C_1', 'BP_Trailer_C_4', 'BP_Military_Car_C_2', 'BP_Military_Car_C_3', 'BP_Military_Car_C_4', 'BP_Military_Car_C_5', 
'BP_Military_Car_C_8', 'BP_Military_Pickup_C_6', 'BP_Military_Pickup_C_8', 'BP_Military_Pickup_C_9', 'BP_Military_Pickup_C_10', 
'BP_Military_Pickup_C_11', 'BP_Military_Car_C_9', 'BP_Military_Car_C_11', 'BP_Military_Pickup_C_15', 'BP_Military_Pickup_C_16', 
'BP_Trailer_C_0', 'BP_Trailer_C_5', 'BP_Trailer_C_6', 'BP_Trailer_C_7', 'BP_Trailer_C_10', 'BP_Trailer_C_12', 'BP_SAM_Launcher_C_1', 
'BP_Military_Car_C_12', 'BP_Military_Car_C_13', 'BP_Military_Car_C_10', 'BP_Military_Car_C_14', 'BP_Military_Car_C_15', 
'BP_Military_Car_C_1', 'BP_Military_Car_C_23', 'BP_Military_Pickup_C_12', 'BP_Military_Pickup_C_13', 'BP_Military_Pickup_C_14', 
'BP_Fighter_jet_2_C_3', 'BP_Fighter_jet_2_C_4', 'BP_Fighter_jet_Tiger_C_3', 'BP_Fighter_jet_Pars_C_3', 'BP_Fighter_jet_Tiger_C_4', 
'Car_C_0', 'Car_C_1', 'Car_C_10', 'Car_C_11', 'Car_C_12', 'Car_C_13', 'Car_C_14', 'Car_C_16', 'Car_C_17', 'Car_C_18', 'Car_C_19', 
'Car_C_2', 'Car_C_20', 'Car_C_22', 'Car_C_23', 'Car_C_24', 'Car_C_25', 'Car_C_26', 'Car_C_27', 'Car_C_28', 'Car_C_29', 'Car_C_3', 
'Car_C_31', 'Car_C_4', 'Car_C_5', 'Car_C_6', 'Car_C_7', 'Car_C_8', 'Car_C_9']

# Check if the color map file exists on your system
if os.path.exists(colormapfile_path):
    with open(colormapfile_path, "r") as readfile:
        color_mapping = json.load(readfile)
    print("File loaded successfully.")
    #print(color_mapping)
    # True if any dictionary keys are missing from the list
    print(color_mapping.keys())
    print(actors)
    any_missing = set(actors) - color_mapping.keys()    
    if len(any_missing) > 0: 
      print("This color map doesnt match your intended objects")
      sys.exit()

else:
    print("File not found. Initialized empty data.")
    
    color_map = {}

    # Set how many unique combinations you want to generate
    number_of_combinations = len(actors)
    unique_combinations = set()

    while len(unique_combinations) < number_of_combinations:
        # Individual numbers can repeat (e.g., 42, 42, 105)
        new_tuple = tuple(random.randint(1, 254) for _ in range(3))
        
        # The set ensures the exact same combination isn't added twice
        unique_combinations.add(new_tuple)

    # Print the results
    color_mapping = dict(zip(actors, unique_combinations))
    #print(color_mapping)
    # Save data directly to a file
    with open(colormapfile_path, "w") as file:
        json.dump(color_mapping, file, indent=4)
  

response = client.request(
    "vset /objects/annotation_colors_json colormap_output.json"
)
print(response)

cam_loc = client.request('vget /camera/0/location')  # force spawn
print(f'Spawned camera location is {cam_loc}')

vehicle = 'BP_Military_Pickup_C_1'

capture_360(client, vehicle, actors, radius=300, height=100, steps=36, outdir="capture")
'''
# Get vehicle location
loc = client.request(f'vget /object/{vehicle}/location')
x, y, z = map(float, loc.split())
print(x, y, z)
print(client.request(f'vget /object/{vehicle}/color'))

# Change Camera 0 to 1280x720 resolution
client.request('vset /camera/0/size 1280 720')

# Position camera
cam_x = x + 600
cam_y = y
cam_z = z + 300

client.request(f'vset /camera/0/location {cam_x} {cam_y} {cam_z}')
client.request('vset /camera/0/rotation -20 180 0')

# Capture RGB
rgb_bytes = client.request('vget /camera/0/lit png')

with open(f'{now}_{vehicle}_rgb.png', 'wb') as f:
    f.write(rgb_bytes)

# Capture segmentation mask
mask_bytes = client.request('vget /camera/0/object_mask png')

with open(f'{now}_{vehicle}_mask.png', 'wb') as f:
    f.write(mask_bytes)

print(f"Saved {now}_{vehicle}_rgb.png and {now}_{vehicle}_mask.png")'''
