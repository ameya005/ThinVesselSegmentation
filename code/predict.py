'''
Predicting on the images

'''

import driveUtils
import patchify
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage.filter import threshold_otsu as totsu
import numpy as np
from skimage.color import rgb2hsv,rgb2lab,rgb2luv


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