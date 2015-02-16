import driveUtils
import patchify
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage.filter import threshold_otsu as totsu
import numpy as np
from skimage.color import rgb2hsv, rgb2lab, rgb2luv

def drive_model(patchsize=(10,10),clusters=100):

    # Training Patches
    img = driveUtils.readimage('../training/images/')
    patches = driveUtils.computePatch(img,size=patchsize)

    # Segmentation Patches
    imgGT = driveUtils.readimage('../training/1st_manual/')
    patchesGT = driveUtils.computePatch(imgGT,size=patchsize)
    # Generate Random numbers

    patchesGreen = driveUtils.computePatch(img, channel=1,size=patchsize)

    greenPatch = []
    greenPatchGT = []

    for key in patchesGreen.keys():
        rnumber = (np.random.sample(15000) * 250000).astype('int')
        greenPatch.extend(patchesGreen[key][rnumber])
        greenPatchGT.extend(patchesGT[key][rnumber])

    greenPatch = driveUtils.flattenlist(greenPatch)
    greenPatch = driveUtils.zscore_norm(greenPatch)

    kmG = MiniBatchKMeans(n_clusters=clusters)
    kmG.fit(greenPatch)

    greenIdx = kmG.predict(greenPatch)

    greenCluster = {}

    for i in range(clusters):
        greenCluster[i] = []

    for i, j in enumerate(greenIdx):
        greenCluster[j].append(greenPatchGT[i].astype('uint16'))

    clusterGtG = {}

    for i in greenCluster.keys():
        if len(greenCluster[i]):
            clusterGtG[i] = np.average(greenCluster[i], axis=0)


    return kmG,clusterGtG

def test_predict(kmG,clusterGtG,location,patchsize=(10,10)):

    test_img = driveUtils.readimage('../test/images/')
    testPatchG = driveUtils.computePatch(test_img, channel=1,size=patchsize)

    if not os.path.exists('../Results/'+str(location)):
        os.makedirs('../Results/'+str(location))    

    for key in test_img.keys():     
        tPatchG = driveUtils.zscore_norm(driveUtils.flattenlist(testPatchG[key]))
        # print "Debug Level 1.2"
        patchGidx = kmG.predict(tPatchG)
        # print "Debug Level 3"
        testimg = patchify.unpatchify(
            np.asarray([clusterGtG[j] for i, j in enumerate(patchGidx)]), (584, 565))


        print "Saving_" + str(key)

        plt.imsave('../Results/'+str(location)+'/' + str(key) + '_G' +
                   '.png', testimg, cmap=cm.gray)

####################################

# Model 1

km,clusterModel = drive_model(patchsize=(10,10),clusters=100)
