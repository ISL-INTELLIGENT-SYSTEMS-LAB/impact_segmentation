import matplotlib.pyplot as plt
import cv2
import os
from shutil import move, copy2
from datetime import datetime
import time
from PIL import Image
import numpy as np
import json



def rename_files(root_directory, destination_directory, postfix_list=["png","jpg","json"]):
    os.makedirs(destination_directory, exist_ok=True)
    for roots, dirs, files in os.walk(root_directory):
        for a_file in files:
            filename, postfix = a_file.split(".")
            if postfix in postfix_list:
                # print(os.path.basename(roots))
                timestamp = os.path.basename(roots).replace("placement_data_", "")
                # print(timestamp)
                dest_file = f'{timestamp}_{a_file}'
                dest_path = os.path.join(destination_directory, dest_file)
                # print(dest_path)
                source_path = os.path.join(roots, a_file)
                move(source_path, dest_path)            


def datasetInfo(version, description, contributor, date=None):
    if date is None:
        
        date = datetime.now()

    year = date.strftime('%Y')
    date_created = date.strftime('%Y/%m/%d %H:%M:%S')

    infoDict = { 'description': description,
                'version': version,
                'year': year,
                'contributor': contributor,
                'date_created': date_created}
    return infoDict


def datasetCategories(categoriesDict):
    cata = []
    for i, supercat in enumerate(categoriesDict):
        for cat in categoriesDict[supercat]:
            cata.append({'id':i+1, 'name':cat, 'super_category':supercat})
    return cata


def datasetImages(img_dir, licName, postFix='.png'):
    image_files = sorted(os.listdir(img_dir))
    image_files = [x for x in image_files if x.endswith(postFix)]
    imgList = []
    
    for i, img in enumerate(image_files):
        with Image.open(os.path.join(img_dir, img)) as pic:
            width, height = pic.size
        date_captured = time.ctime(os.path.getctime(os.path.join(img_dir,img)))
        
        imgList.append({"id": i, "width": width , "height": height , "file_name": img,
                    "license": licName, "date_captured": date_captured})
    return imgList


def get_json_objects(json_file_name, seg_mask_file_name):
    object_list = []
    leg_obj_dict = {}
    base_name = os.path.basename(seg_mask_file_name)
    base_name = base_name.replace('.jpg', '')
    xpos, zpos, rotation = base_name.split('_')[-3:]
    with open(json_file_name, 'r') as f:
        data = json.load(f)
    for camera in data['Cameras']:
        if (data['Cameras'][camera]['xpos'] == int(xpos) and data['Cameras'][camera]['zpos'] == int(zpos)) and data['Cameras'][camera]['rotation'] == int(rotation):
            object_list += data['Cameras'][camera]['objects']
    for legend_obj in data['mask_legend']:
        if legend_obj in object_list:
            leg_obj_dict[legend_obj] = data['mask_legend'][legend_obj]
    return leg_obj_dict


def binaryMask(json_file_name, seg_mask_file_name):
    obj_dict = get_json_objects(json_file_name, seg_mask_file_name)
    img_mask = cv2.imread(seg_mask_file_name)
    colors = np.unique(img_mask.reshape(-1, img_mask.shape[2]), axis=0)
    print(colors)
    print(len(colors))
    print(obj_dict)
    for key, value in obj_dict.items():
        high_value = value+7
        low_value = value-7
        print(key)
        mask = cv2.inRange(img_mask, (low_value, low_value, low_value), (high_value, high_value, high_value))
        #plt.imshow(mask, cmap="Greys")
        #plt.show()
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(contours)


img_mask = "./easter_files/20250617_135147_Camera_0_0_45.jpg"
json_file = "/home/santino/scene_integration/easter_files/20250617_135147_data.json"


binaryMask(json_file, img_mask)

category_list = ['pinkEgg', 'purpleEgg', 'yellowBall7', 'yellowBall3', 'bigBall', 'scrub2', 'scrub1']

super_category_list = {'Egg':['pinkEgg', 'purpleEgg'], 'Ball':['yellowBall7', 'yellowBall3', 'bigBall'], 'Scrubber':['scrub2', 'scrub1']}

lnames = {"url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike 4.0 International"}


version_num = "1.0"
description_val = "FSU easter files test"
contributors = ["Catherine Spooner", "Santino Sini"]
root = "/home/santino/scene_integration/work_dir"
dest_dir = "/home/santino/scene_integration/easter_files"
image_postfix = "png"

datasetTest = datasetInfo(version_num, description_val, contributors)
#print(datasetTest)

#print(datasetImages(dest_dir, lnames, image_postfix))
#print(datasetCategories(super_category_list))


rename_files(root, dest_dir)



""" 
basename = os.path.basename(roots)
timestamp_list = basename.split("_")[2:]
timestamp = f'{timestamp_list[0]}_{timestamp_list[1]}' """