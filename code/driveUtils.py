'''
Main file for the project
'''

import os
import glob
from matplotlib import pyplot as plt
from patchify import patchify
import numpy as np
from scipy.stats import zscore
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

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

def normalize(data):
	mean = np.mean(data)
	stdev = np.std(data)

	ndata = [( (x-x.mean())/np.std(x) ) for x in data]
	return


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
			

def zscore_norm(data):
	ndata = [np.asarray(zscore(x)) for x in data]
	for x in ndata:
		x[np.isnan(x)]=0
		x[np.isinf(x)]=0
	#ndata = [x.tolist() for x in ndata]
	return ndata

def apply_mask(img,mask):
	'''
	Apply mask to the image and return masked image

	'''

	mask = mask/255.
	new_img = img*mask

	return new_img

def seg_eval2(im_pred,im_gt,thres=0.1):
	im_pred = (im_pred>thres).astype('int')
	im_gt = im_gt/255

	acc = accuracy_score(im_gt, im_pred)
	prec = precision_score(im_gt,im_pred)
	reca = recall_score(im_gt,im_pred)
	f1 = f1_score(im_gt,im_pred)

	return acc,prec,reca,f1

def seg_eval_roc(img,gt):
	img = img.ravel()
	gt = gt.ravel() / 255

	fpr,tpr,_ =roc_curve(gt, img)
	roc_auc = auc(fpr, tpr)

	return fpr,tpr,roc_auc

def plot_roc(fpr,tpr,roc_auc,lkey="Ours"):
	
	#plt.figure()
	plt.plot(fpr, tpr, label=str(lkey)+'_ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.grid()
	plt.show()
