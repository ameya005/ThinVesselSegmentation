'''
Main file
'''

import driveUtils
import patchify
from sklearn.cluster import KMeans

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


kmR = KMeans(n_clusters=1000,n_jobs=-1)
kmR.fit(redPatch)

kmG = KMeans(n_clusters=1000,n_jobs=-1)
kmG.fit(greenPatch)

kmB = KMeans(n_clusters=1000,n_jobs=-1)
kmB.fit(bluePatch)


'''

Indexing the patches to cluster and predicting

'''

redIdx = km.predict(redPatch)
greenIdx = km.predict(greenPatch)
blueIdx = km.predict(bluePatch)

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
clusterGt = {}

for i in redCluster.keys():
	clusterGt[i] = np.average(redCluster[i],axis=0)

'''
Test on an image
'''

t25red = redPatch['25']
t25green = greenPatch['25']
t25blue = bluePatch['25']

t25redPred = km.predict(t25red)
t25greenPred = km.predict(t25green)
t25bluePred = km.predict(t25blue)

#recreate images

testimg = np.ndarray((584,565,3))

testimg[:,:,0] = patchify.unpatchify(np.asarray([clusterGt[j] for i,j in enumerate(t25redPred)]),(584,565))
testimg[:,:,1] = patchify.unpatchify(np.asarray([clusterGt[j] for i,j in enumerate(t25greenPred)]),(584,565))
testimg[:,:,2] = patchify.unpatchify(np.asarray([clusterGt[j] for i,j in enumerate(t25bluePred)]),(584,565))

# Create segmentation image



# testimgpatches = driveUtils.flattenlist(patches['25'])
# testimgpred = km.predict(testimgpatches)

# testimg_recreate = patchify.unpatchify(np.asarray([clusterGt[j] for i,j in enumerate(testimgpred)]),(584,565))








