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
from skimage.morphology import binary_erosion, disk,rectangle
import cPickle as pickle
from skimage import exposure
import cv2
import skimage.transform as skt

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

def erode_mask(masks,seradius=6):

	se =disk(seradius)

	for key in masks.keys():
		masks[key] = binary_erosion(masks[key], se)

	return masks

def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axs = plt.subplots(nrows,ncols, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)

    axs = axs.ravel()
    ks = figures.keys()
    for i,j in enumerate(ks):
        axs[i].imshow(clusterGtG[j])
        axs[i].set_xticks([])
        axs[i].set_yticks([])

def plot_compare(modelname):
	mask_img = driveUtils.readimage('../test/mask/')
	gt1_img = driveUtils.readimage('../test/1st_manual/')
	gt2_img = driveUtils.readimage('../test/2nd_manual/')

	img = {}

	file_green =glob.glob("../Results/" +str(modelname) +"/??_G.png")

	for gfile in file_green:
		key = os.path.splitext(gfile)[0][-4:-2]
		img[key] = rgb2gray(plt.imread(gfile)) * mask_img[key]
		img[key] = img[key]/np.max(img[key])

	plotroc(img,gt1_img,modelname)



def erode_mask_new(masks,seradius=6):

	se =rectangle(seradius,seradius)

	for key in masks.keys():
		masks[key] = binary_erosion(masks[key], se)

	return masks

def save_model(filename,keymdl):
	with open(filename,'wb') as fp:
		pickle.dump(keymdl,fp)

def adapteq(img):
	'''
	img : dict
	'''

	for key in img.keys():
		img[key] = exposure.equalize_adapthist(img[key])

	return img


def eq_clahe(img,tilesize=(8,8),channel=1,clplmt=2.0):
	'''
	Contrast Limited Adaptive Histogram Equalization
	Using cv2 CLAHE

	
	Input
	-----

	img : 		dict,	RGB images in form of dictionary
	tilesize:	tuple,	tileGridSize value for CLAHE creation
	clplmt:		float,	value for clipLimit in CLAHE

	Return
	------

	img :		dict,	Equalized images

	'''
	#creating a clahe object 
	clahe = cv2.createCLAHE(clipLimit=clplmt,tileGridSize=tilesize)
	for key in img.keys():
		img[key][:,:,0]=clahe.apply(img[key][:,:,0])
		img[key][:,:,1]=clahe.apply(img[key][:,:,1])
		img[key][:,:,2]=clahe.apply(img[key][:,:,2])

	return img

def scaleimg(img,scale=1):
	'''
	Scale the given image by the scaling fatctor

	Input
	-----
	img:	ndarray,	image
	scale:	float,		scaling value
						defaults : 1 ( no scaling)

	Return
	------

	Scaled image

	'''

	simg = skt.rescale(img, scale)
	return simg

def dictimgscale(imdict,scaling=1):
	'''
	Dict of images to scale

	Input
	-----

	imdict:		dict, 	dictionary of image
	scaling:	float,	scaling factor for the images

	Return
	------
	dict, of scaled images
	'''
	for key in imdict.keys():
		imdict[key] = scaleimg(imdict[key],scale=scaling)
		#imdict[key] = imdict[key].astype('uint8')

	return imdict

def dictresizeimage(imgdict,shape=()):
	'''
	Dict of images to resize

	Input
	-----

	imdict:		dict, 	dictionary of image
	shape:		tuple,	size of image to resize to

	Return
	------
	dict, of reszied images
	'''
	for key in imgdict.keys():
		imgdict[key]  = skt.resize(imgdict[key], shape)

	return imgdict

def clusterimg(clustermodel):
	return

def patchGenerate():
	return

def displayPatch(images,M=1,N=1,patchSize=(3,3)):
	'''
	Input
	-----
	images :	list, list of ndarry images		
	M :			int,Number of rows
	N :			int,Number of columns
	patchSize:	tuple, size of patches to be used for display

	Return
	------
	patchImg:	ndarry, of all images

	[TODO]
	* Check for size
	* Add option for resize
	* Additonal checks
	* Add borders
	'''
	totImages = len(images)
	(patchL,patchH ) = patchSize
	sizeImg = images[0].shape

	if patchSize != sizeImg:
		patchSize = sizeImg

	sizeL = M * patchL
	sizeH = N * patchH

	patchImg = np.ones((sizeL,sizeH))
	idx=0
	for i in range(0,sizeL,patchL):
		for j in range(0,sizeH,patchH):
			if idx>=totImages:
				break
			patchImg[i:i+patchL,j:j+patchH] = images[idx]
			idx += 1

	return patchImg

def cluster_to_list(clus):
	'''
	Provides a method to read cluster dictionary to a list
	'''
	#return clus.viewvalues()
	img_list = []

	for key in clus.keys():
		img_list.append(clus[key])

	return img_list

def km_to_list(clus,shape=(21,21)):
	'''
	Reshape the learn cluster center to images

	'''
	img_list =[]
	clus_cen = clus.cluster_centers_
	for i in clus_cen:
		img_list.append(i.reshape(shape))

	return img_list