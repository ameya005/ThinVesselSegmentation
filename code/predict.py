'''
Predicting on the images

'''

import driveUtils
import patchify
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage.filter import threshold_otsu as totsu
import numpy as np
from skimage.color import rgb2hsv, rgb2lab, rgb2luv
import cPickle

'''
Load Trained model
'''

with open("modelR.data", 'rb') as fp:
    kmR = cPickle.load(fp)

with open("modelG.data", 'rb') as fp:
    kmG = cPickle.load(fp)

with open("modelB.data", 'rb') as fp:
    kmB = cPickle.load(fp)

'''
Read all the images
'''

img = driveUtils.readimage('../training/images/')
imgGT = driveUtils.readimage('../training/1st_manual/')

patchesRed = driveUtils.computePatch(img, channel=0)
patchesGreen = driveUtils.computePatch(img, channel=1)
patchesBlue = driveUtils.computePatch(img, channel=2)

patchesGT = driveUtils.computePatch(imgGT)

'''
Predict Cluster for GT using the enitre training dataset

'''
redCluster = {}
greenCluster = {}
blueCluster = {}
clusterGtR = {}
clusterGtG = {}
clusterGtB = {}

for i in range(1000):
    clusterGtR[i] = []
    clusterGtG[i] = []
    clusterGtB[i] = []

# Red Channel

for key in patchesRed.keys():
    print key
    tPatchR = driveUtils.zscore_norm(driveUtils.flattenlist(patchesRed[key]))
    redIdx = kmR.predict(tPatchR)

    for i in range(1000):
        redCluster[i] = []

    for i, j in enumerate(redIdx):
        redCluster[j].append(patchesGT[key][i].astype('uint16'))

    for i in redCluster.keys():
        if len(redCluster[i]) == 0:
            continue

        clu_list = []
        if len(clusterGtR[i]):
            clu_list.append(clusterGtR[i])
        if len(redCluster[i]):
            clu_list.extend(redCluster[i])
        # clusterGtR[i].extend(redCluster[i])
        if len(clu_list):
            clusterGtR[i] = np.average(np.asarray(clu_list), axis=0)

#Green Channel#
for key in patchesGreen.keys():
    print key
    tPatchG = driveUtils.zscore_norm(driveUtils.flattenlist(patchesGreen[key]))
    greenIdx = kmR.predict(tPatchG)

    for i in range(1000):
        greenCluster[i] = []

    for i, j in enumerate(greenIdx):
        greenCluster[j].append(patchesGT[key][i].astype('uint16'))

    for i in greenCluster.keys():
        if len(greenCluster[i]) == 0:
            continue

        clu_list = []
        if len(clusterGtG[i]):
            clu_list.append(clusterGtG[i])
        if len(greenCluster[i]):
            clu_list.extend(greenCluster[i])
        # clusterGtG[i].extend(greenCluster[i])
        if len(clu_list):
            clusterGtG[i] = np.average(np.asarray(clu_list), axis=0)

#Blue Channel#
for key in patchesBlue.keys():
    print key
    tPatchB = driveUtils.zscore_norm(driveUtils.flattenlist(patchesBlue[key]))
    blueIdx = kmR.predict(tPatchB)

    for i in range(1000):
        blueCluster[i] = []

    for i, j in enumerate(blueIdx):
        blueCluster[j].append(patchesGT[key][i].astype('uint16'))

    for i in blueCluster.keys():
        if len(blueCluster[i]) == 0:
            continue

        clu_list = []
        if len(clusterGtB[i]):
            clu_list.append(clusterGtB[i])
        if len(blueCluster[i]):
            clu_list.extend(blueCluster[i])
        # clusterGtB[i].extend(blueCluster[i])
        if len(clu_list):
            clusterGtB[i] = np.average(np.asarray(clu_list), axis=0)


##########################
#dirStart = "../"+dataset+"/images/"
# Read the images patches and predict
test_img = driveUtils.readimage('../test/images/')

testPatchR = driveUtils.computePatch(test_img, channel=0)
testPatchG = driveUtils.computePatch(test_img, channel=1)
testPatchB = driveUtils.computePatch(test_img, channel=2)

for key in test_img.keys():
    print "Debug Level 1"
    tPatchR = driveUtils.zscore_norm(driveUtils.flattenlist(testPatchR[key]))
    print "Debug Level 1.1"
    tPatchG = driveUtils.zscore_norm(driveUtils.flattenlist(testPatchG[key]))
    print "Debug Level 1.2"
    tPatchB = driveUtils.zscore_norm(driveUtils.flattenlist(testPatchB[key]))
    print "Debug Level 2"
    patchRidx = kmR.predict(tPatchR)
    patchGidx = kmG.predict(tPatchG)
    patchBidx = kmB.predict(tPatchB)

    print "Debug Level 3"
    testimg = np.ndarray((584, 565, 3))

    testimg[:, :, 0] = patchify.unpatchify(
        np.asarray([clusterGtR[j] for i, j in enumerate(patchRidx)]), (584, 565))
    testimg[:, :, 1] = patchify.unpatchify(
        np.asarray([clusterGtG[j] for i, j in enumerate(patchGidx)]), (584, 565))
    testimg[:, :, 2] = patchify.unpatchify(
        np.asarray([clusterGtB[j] for i, j in enumerate(patchBidx)]), (584, 565))

    print "Saving_" + str(key)
    plt.imsave('../Results/5Feb/test/' + str(key) + '_R' +
               '.png', testimg[:, :, 0], cmap=cm.gray)
    plt.imsave('../Results/5Feb/test/' + str(key) + '_G' +
               '.png', testimg[:, :, 1], cmap=cm.gray)
    plt.imsave('../Results/5Feb/test/' + str(key) + '_B' +
               '.png', testimg[:, :, 2], cmap=cm.gray)
    plt.imsave('../Results/5Feb/test/' + str(key) + '.png', testimg / 255)

    del testimg
