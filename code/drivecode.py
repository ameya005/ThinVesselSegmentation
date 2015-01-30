'''
Main file
'''

import driveUtils
import patchify


# Training Patches
img = driveUtils.readimage('../training/images/')
patches = driveUtils.computePatch(img)

# Segmentation Patches
imgGT = driveUtils.readimage('../training/1st_manual/')
patchesGT = driveUtils.computePatch(imgGT)
#Generate Random numbers


#Red Channel
patchesRed = driveUtils.computePatch(img,channel=0)

redPatch = []
redPatchGT=[]

for key in patchesRed.keys():
	rnumber = (np.random.sample(500)*250000).astype('int')
	redPatch.extend(patchesRed[key][rnumber])
	redPatchGT.extend(patchesGT[key][rnumber])

	redPatch = driveUtils.flattenlist(redPatch)

km = KMeans(n_clusters=100,n_jobs=-1)
km.fit(redPatch)

redIdx = km.predict(redPatch)

redCluster = {}

for i in range(100):
	redCluster[i] =[]

for i,j in enumerate(redIdx):
	redCluster[j].append(redPatchGT[i].astype('uint16'))

clusterGt = {}

for i in redCluster.keys():
	clusterGt[i] = np.average(redCluster[i],axis=0)

'''
Test on an image
'''

testimgpatches = driveUtils.flattenlist(patches['25'])
testimgpred = km.predict(testimgpatches)

testimg_recreate = patchify.unpatchify(np.asarray([clusterGt[j] for i,j in enumerate(testimgpred)]),(584,565))





# #Green Channel
# patchesGreen = driveUtils.computePatch(img,channel=1)

# greenPatch = []
# greenPatchGT=[]

# for key in patchesGreen.keys():
# 	rnumber = (np.random.sample(500)*250000).astype('int')
# 	greenPatch.extend(patchesGreen[key][rnumber])
# 	greenPatchGT.extend(patchesGT[key][rnumber])

# 	greenPatch = flattenlist(greenPatch)

# #Blue Channel
# patchesBlue = driveUtils.computePatch(img,channel=2)

# bluePatch = []
# bluePatchGT=[]

# for key in patchesBlue.keys():
# 	rnumber = (np.random.sample(500)*250000).astype('int')
# 	bluePatch.extend(patchesBlue[key][rnumber])
# 	bluePatchGT.extend(patchesGT[key][rnumber])

# 	bluePatch = flattenlist(bluePatch)


