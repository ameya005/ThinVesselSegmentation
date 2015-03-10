'''
Dictionary Learning
'''
from sklearn.decomposition import MiniBatchDictionaryLearning,DictionaryLearning

def dictlearn():

	## Extracting patches
    img = driveUtils.readimage('../training/images/')
    img = driveUtils.dictimgscale(img,scaling=rescale) #Scaling images

    if clahe:
        img = driveUtils.eq_clahe(img,tilesize=(24,24))
     
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
        rnumber = rand.sample(xrange(len(patchesGreen[key])), 10000)
        rnumber.extend(arange(100))
        greenPatch.extend(patchesGreen[key][rnumber])
        greenPatchGT.extend(patchesGT[key][rnumber])

    greenPatch = driveUtils.flattenlist(greenPatch)
    greenPatch = driveUtils.zscore_norm(greenPatch)
    ##################################################################
    # Learn Dictionary

    km = MiniBatchDictionaryLearning(n_jobs=-1,n_components=500)
    km.fit(greenPatch)

    #################################################################
    greenIdx = km.predict(greenPatch)

    greenCluster = {}

    for i in range(clusters):
        greenCluster[i] = []

    for i, j in enumerate(greenIdx):
        greenCluster[j].append(greenPatchGT[i].astype('uint16'))

    clusterGtG = {}

    for i in greenCluster.keys():
        if len(greenCluster[i]):
            clusterGtG[i] = np.average(greenCluster[i], axis=0)