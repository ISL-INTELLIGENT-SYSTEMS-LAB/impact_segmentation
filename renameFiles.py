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
#           renameFiles.py
#
#   Programmers:
#           Catherine Spooner
#           Carley Brinkley
#           
#
#
#  Revision     Date                        Release Comment
#  --------  ----------  ------------------------------------------------------
#    1.0     10/24/2023  Initial Release
#
#  File Description
#  ----------------
#  Currently Geisel gives us files with names like groups(0).png, instances(0).png, and realistic(0).png.
#	The number inside the parenthesis is supposed to be the file number.
#	The files with realistic in the name are the original screenshots of the scenery.
#	The files with groups in the name are the semantic segmentations (class information).
#	The files with instances in the name are the instance segmentations of all the classes.
#	We need to rename these to include the date of when the file was created as well as 
#	add some project specific information.
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
#   get_time_from_meta
#   rename_files
#
'''
#################################################################################################
#Import Statements
#################################################################################################

from shutil import move
import os
from datetime import datetime
import time
import pathlib

#################################################################################################
# Functions
#################################################################################################
def get_time_from_meta(afile):
	""" returns the time that a file was created in YYYYMMDD_HHMMSS format

	Parameters
	_________

	afile : string
	    The full path of the file to get the creation time from


	Returns
	________

	date_caputured : string
	    The time that the file was created in YYYYMMDD_HHMMSS format

	"""
	# os.path.getctime returns the number of seconds past the epoch
	# time.ctime converts the number of seconds past the epoch to a string
	# in the following form: Tue Oct 24 12:35:19 2023
	# time.strptime creates a struct that splits all of the time.ctime string
	# into its constituent parts, allowing us to use datetime to create a datetime
	# object with those parts which can be formatted in the desired format
	date_captured = time.ctime(os.path.getctime(afile))
	date_captured = time.strptime(date_captured, '%a %b %d %H:%M:%S %Y')
	date_captured = datetime(date_captured[0], date_captured[1], date_captured[2], 
		date_captured[3],date_captured[4], date_captured[5]).strftime('%Y%m%d_%H%M%S')
	return(date_captured)

def rename_files(root, project_name, postfix=None, timestamp=None):
	""" renames files for the IMPACT project. Currently Geisel gives us files
	with names like groups(0).png, instances(0).png, and realistic(0).png.
	The number inside the parenthesis is supposed to be the file number.
	The files with realistic in the name are the original screenshots of the scenery.
	The files with groups in the name are the semantic segmentations (class information).
	The files with instances in the name are the instance segmentations of all the classes.
	We need to rename these to include the date of when the file was created as well as 
	add some project specific information.

	Parameters
	_________

	root : str
	    The full path where the list of files are located

	project_name : str
		name that will be added to all of the files to help identify which project
		they belong to

	postfix : str
		image file extention. ex: ".png",which is the default. Can be overridden by
		providing it

	timestamp : str
		timestamp that will be added to each file so that they have a unique name.
		If no timestamp is provided, use the time from the first file in the list of 
		images. 

	Returns
	________

	None

	"""

	#constants that can be changed according to need
	GROUPS = 'groups'
	GROUP_NAME = 'semantic_seg'
	INSTANCES = 'instances'
	INSTANCE_NAME = 'instance_seg'
	REALISTIC = 'realistic'
	OG_IMAGE_NAME = 'realimage'
	POSTFIX = '.png'

	if postfix is None:
		postfix = POSTFIX

	imgList = os.listdir(root)

	# only process image files
	imgList = [x for x in imgList if x.endswith(postfix)]

	if timestamp is None:
		timestamp = get_time_from_meta(os.path.join(root, imgList[0]))

	for img in imgList:
		if project_name not in img:
			img_src = os.path.join(root, img)

			if REALISTIC in img:
				new_name = OG_IMAGE_NAME
			elif GROUPS in img:
				new_name = GROUP_NAME
			elif INSTANCES in img:
				new_name = INSTANCE_NAME

			imgID = img.split('(')[1].split(')')[0]
			dest_filename = f'{project_name}_{timestamp}_{new_name}_{imgID:>04}{postfix}'
			img_dest = os.path.join(root, dest_filename)
			move(img_src, img_dest)


if __name__ == "__main__":
    
	rootdir = '/home/cspooner/IMPACT/few-shot-object-detection/datasets/custom/group2'
	proj_name = "IMPACT_mars_simulation"

	rename_files(rootdir, proj_name)

