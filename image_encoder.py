'''
#################################################################################################
#
#
#   Fayetteville State University Intelligence Systems Laboratory (FSU-ISL)
#
#   Mentors:
#           Dr. Sambit Bhattacharya
#
#   File Name:
#           image_encoder.py
#
#   Programmers:
#           Catherine Spooner
#           Carley Brinkley
#           
#
#
#  Revision     Date                        Release Comment
#  --------  ----------  ------------------------------------------------------
#    1.0     10/30/2023  Initial Release
#
#  File Description
#  ----------------
#  This file is used to encode the semantic and instance segmentation images into a single annotation image.
#  Each pixel in the semenatic segmentation image is converted according to the index defined by their
#  color in the list below and then put in the red channel of the final annotation image.
#  Each pixel of the instance images are converted to an arbitrary index, which is put into the green channel of the
#  final annotation image.
#
#   Define a colormap for our semantic segmentation files
#   index, color, name
#   ____________________

#   1, (0, 0, 0), "bedrock"
#   2, (42,114,60), "sky"
#   3, (89, 89, 89), "sand/ground"
#   4, (255, 0, 0), "ventifact"
#   5, (0,255,0), "science_rock"
#   6, (0, 0, 255), "blueberry"
#
#
#  *Classes/Functions organized by order of appearance
#
#  OUTSIDE FILES REQUIRED
#  ----------------------
#   None
#
#  CLASSES
#  -------
#   None
#
#  FUNCTIONS
#  ---------
#   flip_colors
#   create_image_mask_dictionary
#   encodeImageMask
#   mapImageMask
#   createEncoding
#   main
#
'''
#################################################################################################
#Import Statements
#################################################################################################

# import libraries
import os
import sys
import cv2

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from shutil import copy2

print('Libraries imported')

#################################################################################################
# Functions
#################################################################################################
def flip_colors(color):
    """ flips the b and r channels of a color

    Parameters
    _________

    color : tuple
        (r, g, b) or (b, g, r) tuple of color

    Returns
    ________

    flipped_color : tuple
       the b and r channels reversed. if started with (r, g, b), returns (b, g, r).

    """
    r, g, b = color
    flipped_color = (b, g, r)
    return flipped_color

def create_image_mask_dictionary(segImage, approved_colors=None):
    """ creates a dictionary of colors that map to a binary mask created by those colors

    Parameters
    _________

    segImage : image object
        segmentation file

    Returns
    ________

    collections : dict
       all of the binary masks from the segmentation image.
       The key is the color of the original segmentation object, and the value is the binary mask

    """

    #if the image was opened in opencv, which opens them in bgr, then flip everything

    if hasattr(segImage, 'shape'):
        h, w, _ = segImage.shape

    else:
        w, h = segImage.size

    # extract all of the unique colors
    object_seg = np.array(segImage)
    colors = np.unique(object_seg.reshape(-1, object_seg.shape[2]), axis=0)

    collections = {}

    for k in colors:
        color = tuple(k)
        
        if approved_colors is not None:
          if color not in collections.keys() and color in approved_colors:
              mask = cv2.inRange(object_seg, k, k)
              collections[color] = mask
        else:
          if color not in collections.keys():
              mask = cv2.inRange(object_seg, k, k)
              collections[color] = mask

    return(collections)

def encodeImageMask(segImage, colors2indexMap=None):
    """ encodes an image mask with either the color2index map values in the case of semantic segmentation (class values)
    or an arbitrary index number in the case of instance segmentation. In the instance segmentation case, the index value
    depends on the order in which the color is pulled from the dictionary.

    Parameters
    _________

    segImage : image object
        segmentation file

    colors2indexMap : dictionbary
        the dictionary that maps the semantic segmentation colors to their class id.
        This is an optional parameter. Pass it only in the case of semantic segmentation.
        Do not pass if you are using instance segmentation unless you have mapped every color
        to a particular id that you are interested in using.


    Returns
    ________

    final_result : np.array
        all of the binary masks from a segmentation image, mapped to various indices
    """

    collections = create_image_mask_dictionary(segImage, colors2indexMap)
    results = []

    if colors2indexMap is not None:
        newmap = {}

        if hasattr(segImage, 'shape'):
            for key, value in colors2indexMap.items():
                newkey = flip_colors(key)
                newmap[newkey] = value
        else:
            newmap = colors2indexMap


    for index, color in enumerate(collections.keys()):
        mask = collections[color]
        h, w = np.array(mask).shape

        if colors2indexMap is None:
            r = np.full((h, w), index+1)
        else:
            r = np.full((h, w), newmap[color])

        result = cv2.bitwise_and(r,r, mask= mask)
        results.append(result)


    final_result = results[0]
    for result in results:
        # In this case, we are using a bitwise_or to combine all of the results. This
        # step combine all of the results together.
        final_result = cv2.bitwise_or(final_result,result)

    return final_result

def mapImageMask(colors2indexMap, semSegImage, inSegImage=None):
    """ returns an r, g, b image with the semantic segmentation encoded in the r channel, and the 
    instance segmentation encoded in the g channel. The b channel is all 0s.

    Parameters
    _________

    semSegImage : image object
        semantic segmentation file

    inSegImage : image object
        instance segmentation file

    colors2indexMap: dict
        the dictionary that maps the semantic (class) labels to the class id. This is used in
        the case of semantic segmentation only and never instance segmentation.


    Returns
    ________

    rgb : np.array
        the final rgb image. Its still an array, so needs to be converted from array to image
    """

    # b channel is always 0

    r = encodeImageMask(semSegImage, colors2indexMap) + 1
    
    if inSegImage is not None:
      g = encodeImageMask(inSegImage) + 1
    else:
      g = encodeImageMask(semSegImage, colors2indexMap) + 1
    b = np.zeros_like(r)

    # this last step combines all 3 arrays so that we can create an image from them.
    rgb = np.dstack((r, g, b))
    return rgb

def createEncoding(source_dir, dest_dir, map_col, imDIR='realimage', encodeDIR='seg', segName='semantic_seg', inName='instance_seg', seg_POSTFIX='.png', in_POSTFIX='.png'):
    """ returns an r, g, b image with the semantic segmentation encoded in the r channel, and the 
    instance segmentation encoded in the g channel. The b channel is all 0s.

    Parameters
    _________

    source_dir : image object
        semantic segmentation file

    dest_dir : image object
        instance segmentation file

    map_col: dict
        the dictionary that maps the semantic (class) labels to the class id. This is used in
        the case of semantic segmentation only and never instance segmentation.

    imDIR : str
        optional
        name that you want to call the seperate destination directory for the real image files
    
    encodeDIR : str
        optional
        that you want to call the seperate destination directory for the encoded segmentation files

    segName : str
        optional
        name fragment for the semantatic segmentation files

    inName : str
        optional
        name fragment for the instance segmentation files

    Returns
    ________

    None

    """
    image_destdir = os.path.join(dest_dir, imDIR)
    encoded_segdestdir = os.path.join(dest_dir, encodeDIR)
    # create the encoded seg directory if it doesnt exist
    os.makedirs(image_destdir, exist_ok=True)
    os.makedirs(encoded_segdestdir, exist_ok=True)

    # get list of segmentation files to process
    all_files = os.listdir(source_dir)
    # make sure that only png files are being process. Should probably change this to check
    # for valid images, not just png files.
    semseg_files = [x for x in all_files if x.endswith(seg_POSTFIX) and segName in x]
    inseg_files = [x for x in all_files if x.endswith(in_POSTFIX) and inName in x]
    total_files = len(semseg_files)

    for i, afile in enumerate(semseg_files):

        print(f"processing {afile}: {i+1} out of {total_files}")
        semseg_filepath = os.path.join(source_dir, afile) # full path of the image to be processed
        inseg_filepath = os.path.join(source_dir, afile.replace(segName, inName)) # full path of the image to be processed

        realName = afile.replace(segName, "realimage")
        encodedsegName = afile.replace(segName, "encoded_seg")
        semseg_img = cv2.imread(semseg_filepath)
        inseg_img = cv2.imread(inseg_filepath)
        encoded_seg = mapImageMask(map_col, semseg_img, inseg_img) # create new encoded segmentation image
        encoded_seg = Image.fromarray(encoded_seg.astype('uint8'), 'RGB') # create a PIL image from np array
        encoded_seg.save(os.path.join(encoded_segdestdir, encodedsegName)) # save the image to disk
        copy2(os.path.join(source_dir, realName), os.path.join(image_destdir, realName))

def main():
    # Define a colormap for our situation : Still need to deal with the sky
    # for now any color that isnt in this list is assumed to be sky
    colors2index = {(0, 0, 0):1, (89, 89, 89):2, (255, 0, 0):3, (0,255,0):4, (0, 0, 255):5}

    #name2colors = {"bedrock":(0, 0, 0), "sand/ground":(89, 89, 89), 
                    #"ventifact":(255, 0, 0), "science_rock":(0,255,0), "blueberry":(0, 0, 255)}


    user = os.getlogin()

    root_dir = os.path.join(f'/home/{user}/IMPACT/Data_Collection/Mars_Simulation')
    imagedir = os.path.join(root_dir, 'ImgDataSet6')
  
    semseg = 'semantic_seg'
    inseg = "instance_seg"

    createEncoding(imagedir, root_dir, colors2index)

if __name__ == "__main__":
    main()
