'''
One image test
'''
import driveUtils
import patchify
from scipy.stats import zscore
from sklearn.cluster import KMeans

#Load an image
img = plt.imread('../training/images/23_training.tif')
imgGT = plt.imread('../training/1st_manual/23_manual1.gif')

#Seperate R,G,B
imgR = img[:,:,0]
imgG = img[:,:,1]
imgB = img[:,:,2]



#Extract Patches 

patch = patchify.patchify(imgR,(16,16))[0]
patchGT = patchify.patchify(imgGT,(16,16))[0]

'''
Store patches in the working directory
'''

testDict={}
testDict['1']=patch

#storePatch(testDict,dataset="testing",nomrmalize="no")


'''
Vectorize The patches
'''

vPatch = driveUtils.flattenlist(patch)
vPatchGT = driveUtils.flattenlist(patchGT)

'''
Normalize the patch
'''
nPatch = np.asarray([zscore(x) for x in vPatch])
nPatchGT = np.asarray([zscore(x) for x in vPatchGT])

'''
Clustering
'''	

km = KMeans(n_clusters=100,n_jobs=-1)

#Fit the datas
km.fit(nPatch)
#Index of each patc
indx = km.predict(nPatch)

#For the groundtruth
