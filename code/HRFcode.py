'''
HRF image dataset
'''
import driveUtils
import patchify
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage.filter import threshold_otsu as totsu
import numpy as np
from skimage.color import rgb2hsv, rgb2lab, rgb2luv
import random as rand
import os
from joblib import Parallel, delayed  
import multiprocessing
from skimage.transform import resize
from sklearn.cross_validation import train_test_split

# Read the images
def readimage(dir):
    '''
    Read image changed to match STARE dataset naming
    '''

    files = os.listdir(dir)
    img = {}

    for file in files:
        img[os.path.splitext(file)[0]] = plt.imread(dir+file)

    return img

def imagesplit():
    '''
    Split the images into training and test set

    '''
    #Images
    img = readimage('../dataset/images/')
    # GT Images
    imgGT = readimage('../dataset/gt/')
    #img mask
    mask = readimage('../dataset/mask/')

    # Conver the dictionary to Structure array
    imgnd = np.array(img.items(),dtype=dtype)
    imgGTnd = np.array(imgGT.items(),dtype=dtype)
    masknd = np.array(mask.items(),dtype=dtype)

    # Sort the structured arrays so the split key are same
    imgnd = imgnd[imgnd[:,0].argsort()]
    imgGTnd = imgGTnd[imgGTnd[:,0].argsort()]
    masknd = masknd[masknd[:,0].argsort()]

    #Train Test split
    img_train,img_test,gt_train,gt_test,mask_train,mask_test = train_test_split(imgnd,imgGTnd,masknd,test_size=0.6,random_state=42)

    #Convert the structured array to dict to reuse old code
    img_train = dict(img_train)
    img_test = dict(img_test)
    gt_train = dict(gt_train)
    gt_test = dict(gt_test)
    mask_train = dict(mask_train)
    mask_test = dict(mask_test)

    return img_train,gt_train,mask_train,img_test,gt_test,mask_test