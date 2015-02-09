'''
Predicting on the images

'''

import driveUtils
import patchify
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage.filter import threshold_otsu as totsu
import numpy as np
from skimage.color import rgb2hsv,rgb2lab,rgb2luv
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

patchesRed = driveUtils.computePatch(img,channel=0)
patchesGreen = driveUtils.computePatch(img,channel=1)
patchesBlue = driveUtils.computePatch(img,channel=2)

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
	clusterGtR[i] = [0]*256
	clusterGtG[i] = [0]*256
	clusterGtB[i] = [0]*256

for key in patchesRed.keys():
	tPatchR = driveUtils.zscore_norm(driveUtils.flattenlist(patchesRed[key]))
	redIdx = kmR.predict(tPatchR)

	for i in range(1000):
		redCluster[i] =[]
		greenCluster[i] =[]
		blueCluster[i] =[]

	for i,j in enumerate(redIdx):
		redCluster[j].append(patchesGT[key][i].astype('uint16'))
		

	for i in redCluster.keys():
		clusterGtR[i].extend(redCluster[i])
		clusterGtR[i] = np.average(np.asarray(clusterGtR[i]),axis=0)

greenIdx = kmG.predict(greenPatch)
blueIdx = kmB.predict(bluePatch)





for i,j in enumerate(greenIdx):
	greenCluster[j].append(greenPatchGT[i].astype('uint16'))

for i,j in enumerate(blueIdx):
	blueCluster[j].append(bluePatchGT[i].astype('uint16'))


'''

Groudn truth clustering
'''




for i in greenCluster.keys():
	clusterGtG[i] = np.average(greenCluster[i],axis=0)

for i in blueCluster.keys():
	clusterGtB[i] = np.average(blueCluster[i],axis=0)

##########################
#dirStart = "../"+dataset+"/images/"
#Read the images patches and predict
test_img = driveUtils.readimage('../test/images/')

testPatchR = driveUtils.computePatch(test_img,channel=0)
testPatchG = driveUtils.computePatch(test_img,channel=1)
testPatchB = driveUtils.computePatch(test_img,channel=2)

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
	testimg = np.ndarray((584,565,3))

	testimg[:,:,0] = patchify.unpatchify(np.asarray([clusterGtR[j] for i,j in enumerate(patchRidx)]),(584,565))
	testimg[:,:,1] = patchify.unpatchify(np.asarray([clusterGtG[j] for i,j in enumerate(patchGidx)]),(584,565))
	testimg[:,:,2] = patchify.unpatchify(np.asarray([clusterGtB[j] for i,j in enumerate(patchBidx)]),(584,565))

	print "Saving_" +str(key)
	plt.imsave('../Results/5Feb/test/'+str(key)+'_R' + '.png',testimg[:,:,0],cmap=cm.gray)
	plt.imsave('../Results/5Feb/test/'+str(key)+'_G' + '.png',testimg[:,:,1],cmap=cm.gray)
	plt.imsave('../Results/5Feb/test/'+str(key)+'_B' + '.png',testimg[:,:,2],cmap=cm.gray)
	plt.imsave('../Results/5Feb/test/'+str(key)+ '.png',testimg/255)

	del testimg