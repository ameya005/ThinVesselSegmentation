'''
Main file for the project
'''

import os
import glob
from matplotlib import pyplot as plt
from patchify import patchify
import numpy as np

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

def normalize(data):
	mean = np.mean(data)
	stdev = np.std(data)

	ndata = [( (x-x.mean())/np.std(x) ) for x in data]
	return

def storePatch(imgPatch,dataset="training",normalize="no"):
	'''
	Stores the patch on the disk

	Creates the directory based on the key. Stores the images
	'''
	dirStart = "../"+dataset+"/images/"

	for key in imgPatch.keys():
		if not os.path.exists(dirStart+key):
			os.makedirs(dirStart+key)

		for i,j in enumerate(imgPatch[key]):
			plt.imsave(dirStart+key+'/'+str(i)+'.png',j,cmap=cm.gray)

def readPatch(keydir,dataset="training"):
	'''
	Read the stored patches
	'''
	dirStart = "../"+dataset+"/images/"

	keys = os.listdir(dirStart)

	
	for key in keys:
		dirNew = dirStart+key
			
