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

#Green Channel
patchesGreen = driveUtils.computePatch(img,channel=1)

greenPatch = []
greenPatchGT=[]
for key in patchesGreen.keys():
	rnumber = (np.random.sample(500)*250000).astype('int')
	greenPatch.extend(patchesGreen[key][rnumber])
	greenPatchGT.extend(patchesGT[key][rnumber])

#Blue Channel
patchesBlue = driveUtils.computePatch(img,channel=0)

bluePatch = []
bluePatchGT=[]
for key in patchesBlue.keys():
	rnumber = (np.random.sample(500)*250000).astype('int')
	bluePatch.extend(patchesBlue[key][rnumber])
	bluePatchGT.extend(patchesGT[key][rnumber])