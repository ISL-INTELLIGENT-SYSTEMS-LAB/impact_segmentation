import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def stack_images(dirList, imgpostfix):

  imgList = []
  
  for dirs in dirList:
    imgNames = os.listdir(dirs)
    imgNames = [os.path.join(dirs,x) for x in imgNames if x.endswith(imgpostfix)]
    imgList.extend(imgNames)
  
  for imgPath in imgList:
    pic = Image.open(imgPath)
    if len(pic.size) < 3:
      pic_arr = np.array(pic)
      pic_arr = np.dstack((pic_arr, pic_arr, pic_arr))
      img = Image.fromarray(pic_arr.astype('uint8'), 'RGB') # create a PIL image from np array
      img.save(imgPath) # save the image to disk
  
  
def get_mean_std(dirList, imgpostfix):

  imgList = []
  
  for dirs in dirList:
    imgNames = os.listdir(dirs)
    imgNames = [os.path.join(dirs,x) for x in imgNames if x.endswith(imgpostfix)]
    imgList.extend(imgNames)
  
  mean_r = []
  mean_g = []
  mean_b = []
  std_r = []
  std_g = []
  std_b = []
  
  for imgPath in imgList:
    #imgPath = os.path.join(directory, img)
    pic = Image.open(imgPath)
    pic_arr = np.array(pic)
    #pic_arr = np.dstack((pic_arr, pic_arr, pic_arr))
    
    mean_r.append(np.mean(pic_arr[:,:,0]))
    mean_g.append(np.mean(pic_arr[:,:,1]))
    mean_b.append(np.mean(pic_arr[:,:,2]))
    
    std_r.append(np.std(pic_arr[:,:,0]))
    std_g.append(np.std(pic_arr[:,:,1]))
    std_b.append(np.std(pic_arr[:,:,2]))
    
  mean = (np.mean(mean_r), np.mean(mean_g), np.mean(mean_b))
  std = (np.std(std_r), np.std(std_g), np.std(std_b))
  return(mean, std)

dlist = ["ai4mars-dataset-merged-0.1/re_encoded_files/images_test", "ai4mars-dataset-merged-0.1/re_encoded_files/images_train"]
imgpost = ".JPG"

stack_images(dlist, imgpost)
#print(get_mean_std(dlist, imgpost))