import matplotlib.pyplot as plt
import cv2
import os
from shutil import move, copy2
from datetime import datetime
import time
from PIL import Image



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

print(datasetImages(dest_dir, lnames, image_postfix))
#print(datasetCategories(super_category_list))


rename_files(root, dest_dir)



""" 
basename = os.path.basename(roots)
timestamp_list = basename.split("_")[2:]
timestamp = f'{timestamp_list[0]}_{timestamp_list[1]}' """