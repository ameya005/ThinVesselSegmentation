'''
Main file for the project
'''

import os
import glob
from matplotlib import pyplot as plt
from patchify import patchify

def readimage(dir):
	'''	Read the image and decompose into different color channels

	
	Input
	-----
	dir -- the absolute/relative address of the directory to read from

	Output
	------
	img -- ditionary of images
	'''
	files = os.listdir(dir)
	img = {}

	for file in files:
		img[os.path.splitext(file)[0][:2]] = plt.imread(dir+file)

	return img

def computePatch(img,size=(16,16),channel=0):
	''' Compute patches for each image

	Input
	-----
	img     -- dict of images
	size    -- size of the patch
	channel -- Channel number 0:R 1:G 2:B

	Output
	------
	imgPatch : dict with patches

	'''

	imgPatch = {}
	try:
		for key in img.keys():
			imgPatch[key] = patchify(img[key][:,:,channel],size)[0]
	except IndexError:
		for key in img.keys():
			imgPatch[key] = patchify(img[key],size)[0]		
	
	return imgPatch

def flattenarray(arr):
	'''
	Flatten the array
	'''
	farr = np.ndarray.flatten(arr)
	return farr

def flattenlist(li):
	'''
	Flatten all arrays in the list
	'''
	
	fli = [flattenarray(x) for x in li]
	return fli

def cluster(data):
	return

def whiten():
	return

def normalize():
	return