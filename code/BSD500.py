'''
Bekley Dataset Segmentation
'''
import scipy.io
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
import cPickle as pickle


def readimage(dir,size=(481,321)):
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
		im =plt.imread(dir+file)
		if im.shape[:2] != shape:
			im = im.transpose(1,0,2)
		img[os.path.splitext(file)[0]] = im

	return img


def readGT(dir,size=(481,321)):
	'''
	Reading mat files for Ground Truth

	'''
	files = os.listdir(dir)
	img = {}

	for file in files:
		mat = scipy.io.loadmat(dir+file)
		im = mat['groundTruth'][0][0][0][0][1]
		if im.shape != size:
			im = im.T

		img[os.path.splitext(file)[0]] = im

	return img

# Read the images
trainIMG = readimage('/home/kkhandel/Kushal/Projects/sem2/drivedataset/Datasets/BerkleySegment/BSR/BSDS500/data/images/train/')
trainGT  = readGT('/home/kkhandel/Kushal/Projects/sem2/drivedataset/Datasets/BerkleySegment/BSR/BSDS500/data/groundTruth/train/')

def bsd_model(patchsize=(10,10),clusters=100,clahe=False,rescale=1):
	'''
	BSDS500 Dataset 
	'''
    # Training Patches
    img = readimage('/home/kkhandel/Kushal/Projects/sem2/drivedataset/Datasets/BerkleySegment/BSR/BSDS500/data/images/train/')
    # Segmentation Patches
    imgGT = readGT('/home/kkhandel/Kushal/Projects/sem2/drivedataset/Datasets/BerkleySegment/BSR/BSDS500/data/groundTruth/train/')

    patchesGT = driveUtils.computePatch(imgGT,size=patchsize)
    # Generate Random numbexs

    patchesRed = driveUtils.computePatch(img, channel=0,size=patchsize)
    patchesGreen = driveUtils.computePatch(img, channel=1,size=patchsize)
    patchesBlue = driveUtils.computePatch(img, channel=2,size=patchsize)

    redPatch = []
    greenPatch = []
    bluePatch = []
    PatchGT = []

    for key in patchesGreen.keys():
        # rnumber = (np.random.sample(15000) * 250000).astype('int')
        rnumber = rand.sample(xrange(len(patchesGreen[key])), 10000)
        rnumber.extend(arange(100))
        greenPatch.extend(patchesGreen[key][rnumber])
        redPatch.extend(patchesRed[key][rnumber])
        bluePatch.extend(patchesBlue[key][rnumber])
        
        PatchGT.extend(patchesGT[key][rnumber])

    # Normalize Data

    greenPatch = driveUtils.flattenlist(greenPatch)
    greenPatch = driveUtils.zscore_norm(greenPatch)

    redPatch = driveUtils.flattenlist(redPatch)
    redPatch = driveUtils.zscore_norm(redPatch)

    bluePatch = driveUtils.flattenlist(bluePatch)
    bluePatch = driveUtils.zscore_norm(bluePatch)    

    # Fit Data

    kmG = MiniBatchKMeans(n_clusters=clusters, batch_size=10000,verbose=2,reassignment_ratio=0.0001)
    kmR = MiniBatchKMeans(n_clusters=clusters, batch_size=10000,verbose=2,reassignment_ratio=0.0001)
    kmB = MiniBatchKMeans(n_clusters=clusters, batch_size=10000,verbose=2,reassignment_ratio=0.0001)
    
    kmG.fit(greenPatch)
    kmR.fit(redPatch)
    kmB.fit(bluePatch)

    # Predict Indexes
    redIdx = kmG.predict(redPatch)
    greenIdx = kmG.predict(greenPatch)
    blueIdx = kmG.predict(bluePatch)

    # Predict Cluster Red

    redCluster = {}

    for i in range(clusters):
        redCluster[i] = []

    for i, j in enumerate(redIdx):
        redCluster[j].append(PatchGT[i].astype('uint16'))

    clusterGtR = {}

    for i in redCluster.keys():
        if len(redCluster[i]):
            clusterGtR[i] = np.average(redCluster[i], axis=0)

     # Predict Cluster Green
    greenCluster = {}

    for i in range(clusters):
        greenCluster[i] = []

    for i, j in enumerate(greenIdx):
        greenCluster[j].append(PatchGT[i].astype('uint16'))

    clusterGtG = {}

    for i in greenCluster.keys():
        if len(greenCluster[i]):
            clusterGtG[i] = np.average(greenCluster[i], axis=0)

     # Predict Cluster Blue
    blueCluster = {}

    for i in range(clusters):
        blueCluster[i] = []

    for i, j in enumerate(blueIdx):
        blueCluster[j].append(PatchGT[i].astype('uint16'))

    clusterGtB = {}

    for i in blueCluster.keys():
        if len(blueCluster[i]):
            clusterGtB[i] = np.average(blueCluster[i], axis=0)


    return kmG,clusterGtG

def test_predict(kmG,clusterGtG,location,patchsize=(10,10)):

    test_img = readimage('/home/kkhandel/Kushal/Projects/sem2/drivedataset/Datasets/BerkleySegment/BSR/BSDS500/data/images/train/')

    a,b,c = test_img.values()[1].shape
    pre_imgs ={}
    testPatchG = driveUtils.computePatch(test_img, channel=1,size=patchsize)

    if not os.path.exists('../ResultsBSD/'+str(location)):
        os.makedirs('../ResultsBSD/'+str(location))    

    for key in test_img.keys():     
        tPatchG = driveUtils.zscore_norm(driveUtils.flattenlist(testPatchG[key]))
        # tPatchG = driveUtils.flattenlist(testPatchG[key])
        # print "Debug Level 1.2"
        patchGidx = (kmG.predict(tPatchG)).astype('uint8')
        # print "Debug Level 3"
        testimg = patchify.unpatchify(
            np.asarray([clusterGtG[j] for i, j in enumerate(patchGidx)]), (a,b))

        #testimg = resize(testimg, (584,565))

        print "Saving_" + str(key)
        print testimg.shape
        plt.imsave('../ResultsBSD/'+str(location)+'/' + str(key) +
                   '.png', testimg,format='png')

        pre_imgs[key] = testimg

    with open('ontrain.data', 'wb') as fp:
		pickle.dump(pre_imgs, fp)


