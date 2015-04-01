'''
Dictionary Learning
'''
import os
import glob
from matplotlib import pyplot as plt
from patchify import patchify
import numpy as np
from scipy.stats import zscore
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from skimage.morphology import binary_erosion, disk,rectangle,binary_closing
import cPickle as pickle
from skimage import exposure
import cv2
import skimage.transform as skt
from sklearn.decomposition import MiniBatchDictionaryLearning,DictionaryLearning
import random
import driveUtils

def dictlearn(patchsize=(10,10),clusters=100,clahe=False,rescale=1,thres = 0):

	## Extracting patches
    img = driveUtils.readimage('../training/images/')
    img = driveUtils.dictimgscale(img,scaling=rescale) #Scaling images

    # Segmentation Patches
    imgGT = driveUtils.readimage('../training/1st_manual/')
    imgGT = driveUtils.dictimgscale(imgGT,scaling=rescale) # Scaling

    patchesGT = driveUtils.computePatch(imgGT,size=patchsize)
    # Generate Random numbexs

    patchesGreen = driveUtils.computePatch(img, channel=1,size=patchsize)

    greenPatch = []
    greenPatchGT = []

    for key in patchesGreen.keys():
        # rnumber = (np.random.sample(15000) * 250000).astype('int')
        rnumber = random.sample(xrange(len(patchesGreen[key])), 10000)
        rnumber.extend(arange(100))
        greenPatch.extend(patchesGreen[key][rnumber])
        greenPatchGT.extend(patchesGT[key][rnumber])

    greenPatch = driveUtils.flattenlist(greenPatch)
    greenPatch = driveUtils.zscore_norm(greenPatch)
   
   	#Ground truth 
    greenPatchGT = driveUtils.flattenlist(greenPatchGT)
    #greenPatchGT = driveUtils.zscore_norm(greenPatchGT)
    ##################################################################
    # Learn Dictionary

    #km = MiniBatchDictionaryLearning(n_jobs=-1,n_components=500,fit_algorithm='cd',transform_algorithm='lasso_cd',verbose=2,random_state=42)
    km = MiniBatchDictionaryLearning(n_jobs=-1,n_components=clusters,fit_algorithm='cd',transform_algorithm='threshold',verbose=2,random_state=42,transform_alpha=0.2)

    km.fit(greenPatch)

    #################################################################
    greenIdx = km.transform(greenPatch)

    # Work around
    
    idx = (greenIdx > thres).astype(int)
    
    # sum of the idx ( gives number of elements used to construct the dictGt)
    idxSum = idx.T.sum(axis=1)


    groundPatch = np.array(greenPatchGT)

    dictComponents = km.components_
    
    # Dict Ground is obtained here

    dictGround = np.dot(greenIdx.T,groundPatch)
    # Normalize the dictGround 
    groundDict = dictGround / idxSum

    return km,groundDict

def test_predict(km,groundDict,location,patchsize=(10,10),rescale=1,clahe=False,thres=0):

    test_img = driveUtils.readimage('../test/images/')
    test_img = driveUtils.dictimgscale(test_img,scaling=1) #Scaling
    

    
    #Determine Size of image
    a,b,c = test_img['11'].shape

    testPatchG = driveUtils.computePatch(test_img, channel=1,size=patchsize)

    if not os.path.exists('../Results/'+str(location)):
        os.makedirs('../Results/'+str(location))

    for key in test_img.keys():     
        tPatchG = driveUtils.zscore_norm(driveUtils.flattenlist(testPatchG[key]))
        # tPatchG = driveUtils.flattenlist(testPatchG[key])
        # print "Debug Level 1.2"
        patchGidx = (km.transform(tPatchG))
        patchGidx = (patchGidx >thres).astype(int)
        
        # print "Debug Level 3"
        testimg = np.dot(patchGidx,groundDict)
        testimg = testimg / (patchGidx.sum(axis=1))

        img = patchify.unpatchify(testimg,(a,b))

        #testimg = resize(testimg, (584,565))

        print "Saving_" + str(key)

        plt.imsave('../Results/'+str(location)+'/' + str(key) + '_G' +
                   '.png', img, cmap=cm.gray)

with open('dictModel', 'wb') as fp:
    pickle.dump([km,dictGround], fp)