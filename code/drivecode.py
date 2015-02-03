'''
Main file
'''

import driveUtils
import patchify
from sklearn.cluster import KMeans, MiniBatchKMeans

# Training Patches
img = driveUtils.readimage('../training/images/')
patches = driveUtils.computePatch(img)

# Segmentation Patches
imgGT = driveUtils.readimage('../training/1st_manual/')
patchesGT = driveUtils.computePatch(imgGT)
#Generate Random numbers


#------------------------------------------------------------------------#
'''
Patch Generation
'''
#Red Channel
patchesRed = driveUtils.computePatch(img,channel=0)

redPatch = []
redPatchGT=[]

for key in patchesRed.keys():
	rnumber = (np.random.sample(15000)*250000).astype('int')
	redPatch.extend(patchesRed[key][rnumber])
	redPatchGT.extend(patchesGT[key][rnumber])

	redPatch = driveUtils.flattenlist(redPatch)


#Green Channel
patchesGreen = driveUtils.computePatch(img,channel=1)

greenPatch = []
greenPatchGT=[]

for key in patchesGreen.keys():
	rnumber = (np.random.sample(15000)*250000).astype('int')
	greenPatch.extend(patchesGreen[key][rnumber])
	greenPatchGT.extend(patchesGT[key][rnumber])

	greenPatch = driveUtils.flattenlist(greenPatch)

#Blue Channel
patchesBlue = driveUtils.computePatch(img,channel=2)

bluePatch = []
bluePatchGT=[]

for key in patchesBlue.keys():
	rnumber = (np.random.sample(15000)*250000).astype('int')
	bluePatch.extend(patchesBlue[key][rnumber])
	bluePatchGT.extend(patchesGT[key][rnumber])

	bluePatch = driveUtils.flattenlist(bluePatch)
#----------------------------------------------------------------------------#

'''
Clustering the patches

'''


kmR = MiniBatchKMeans(n_clusters=1000)
kmR.fit(redPatch)

kmG = MiniBatchKMeans(n_clusters=1000)
kmG.fit(greenPatch)

kmB = MiniBatchKMeans(n_clusters=1000)
kmB.fit(bluePatch)


'''

Indexing the patches to cluster and predicting

'''

redIdx = kmR.predict(redPatch)
greenIdx = kmG.predict(greenPatch)
blueIdx = kmB.predict(bluePatch)

redCluster = {}
greenCluster = {}
blueCluster = {}

for i in range(1000):
	redCluster[i] =[]
	greenCluster[i] =[]
	blueCluster[i] =[]

for i,j in enumerate(redIdx):
	redCluster[j].append(redPatchGT[i].astype('uint16'))

for i,j in enumerate(greenIdx):
	greenCluster[j].append(greenPatchGT[i].astype('uint16'))

for i,j in enumerate(blueIdx):
	blueCluster[j].append(bluePatchGT[i].astype('uint16'))


'''

Groudn truth clustering
'''
clusterGtR = {}
clusterGtG = {}
clusterGtB = {}

for i in redCluster.keys():
	clusterGtR[i] = np.average(redCluster[i],axis=0)

for i in greenCluster.keys():
	clusterGtG[i] = np.average(greenCluster[i],axis=0)

for i in blueCluster.keys():
	clusterGtB[i] = np.average(blueCluster[i],axis=0)
'''
Test on an image
'''

t25red = driveUtils.flattenlist(patchesRed['25'])
t25green = driveUtils.flattenlist(patchesGreen['25'])
t25blue = driveUtils.flattenlist(patchesBlue['25'])

t25redPred = kmR.predict(t25red)
t25greenPred = kmG.predict(t25green)
t25bluePred = kmB.predict(t25blue)

#recreate images

testimg = np.ndarray((584,565,3))

testimg[:,:,0] = patchify.unpatchify(np.asarray([clusterGtR[j] for i,j in enumerate(t25redPred)]),(584,565))
testimg[:,:,1] = patchify.unpatchify(np.asarray([clusterGtG[j] for i,j in enumerate(t25greenPred)]),(584,565))
testimg[:,:,2] = patchify.unpatchify(np.asarray([clusterGtB[j] for i,j in enumerate(t25bluePred)]),(584,565))

# Create segmentation image



# testimgpatches = driveUtils.flattenlist(patches['25'])
# testimgpred = km.predict(testimgpatches)

# testimg_recreate = patchify.unpatchify(np.asarray([clusterGt[j] for i,j in enumerate(testimgpred)]),(584,565))








