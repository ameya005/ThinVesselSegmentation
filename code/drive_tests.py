import driveUtils
import patchify
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage.filter import threshold_otsu as totsu
import numpy as np
from skimage.color import rgb2hsv, rgb2lab, rgb2luv
import random as rand
import os
from joblib import Parallel, delayed  
import multiprocessing
from skimage.transform import resize

def drive_model(patchsize=(10,10),clusters=100,clahe=False,rescale=1):

    # Training Patches
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

    kmG = MiniBatchKMeans(n_clusters=clusters, batch_size=10000,verbose=2,reassignment_ratio=0.0001)
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

def test_predict(kmG,clusterGtG,location,patchsize=(10,10),rescale=1,clahe=False):

    test_img = driveUtils.readimage('../test/images/')
    test_img = driveUtils.dictimgscale(test_img,scaling=1) #Scaling
    
    if clahe:
        test_img = driveUtils.eq_clahe(test_img,tilesize=(24,24))
    
    #Determine Size of image
    a,b,c = test_img['11'].shape

    testPatchG = driveUtils.computePatch(test_img, channel=1,size=patchsize)

    if not os.path.exists('../Results/'+str(location)):
        os.makedirs('../Results/'+str(location))    

    for key in test_img.keys():     
        tPatchG = driveUtils.zscore_norm(driveUtils.flattenlist(testPatchG[key]))
        # tPatchG = driveUtils.flattenlist(testPatchG[key])
        # print "Debug Level 1.2"
        patchGidx = (cl1.predict(tPatchG)).astype('uint8')
        # print "Debug Level 3"
        testimg = patchify.unpatchify(
            np.asarray([clusterGtG[j] for i, j in enumerate(patchGidx)]), (a,b))

        #testimg = resize(testimg, (584,565))

        print "Saving_" + str(key)

        plt.imsave('../Results/'+str(location)+'/' + str(key) + '_G' +
                   '.png', testimg, cmap=cm.gray)

def par_predict(key):
    '''
    Parallel Save the code
    '''
    tPatchG = driveUtils.zscore_norm(driveUtils.flattenlist(testPatchG[key]))
    # print "Debug Level 1.2"
    patchGidx = kmG.predict(tPatchG)
    # print "Debug Level 3"
    testimg = patchify.unpatchify(
        np.asarray([clusterGtG[j] for i, j in enumerate(patchGidx)]), (584, 565))


    # print "Saving_" + str(key)

    # plt.imsave('../Results/'+str(location)+'/' + str(key) + '_G' +
    #            '.png', testimg, cmap=cm.gray)

    return testimg


def par_test_predict(kmG,clusterGtG,location,patchsize=(10,10)):
    test_img = driveUtils.readimage('../test/images/')
    global testPatchG
    testPatchG = driveUtils.computePatch(test_img, channel=1,size=patchsize)

    if not os.path.exists('../Results/'+str(location)):
        os.makedirs('../Results/'+str(location))  

    res =Parallel(n_jobs=num_cores)(delayed(par_predict)(key) for key in test_img.keys())

    return res
    
####################################

# Model 1
# Detals : Patch Size = 10 Cluster =100
num_cores = multiprocessing.cpu_count()
km,clusterModel = drive_model(patchsize=(10,10),clusters=50)
test_predict(km,clusterModel,"Model_10_50",patchsize=(10, 10))

# Model 2
# Detals : Patch Size = 10 Cluster =50
km,clusterModel = drive_model(patchsize=(10,10),clusters=100)
test_predict(km,clusterModel,"Model_10_100",patchsize=(10, 10))

# Model 3
# Detals : Patch Size = 10 Cluster =250
km,clusterModel = drive_model(patchsize=(10,10),clusters=250)
test_predict(km,clusterModel,"Model_10_250",patchsize=(10, 10))

km,clusterModel = drive_model(patchsize=(10,10),clusters=500)
test_predict(km,clusterModel,"Model_10_500",patchsize=(10, 10))

km,clusterModel = drive_model(patchsize=(10,10),clusters=750)
test_predict(km,clusterModel,"Model_10_750",patchsize=(10, 10))

km,clusterModel = drive_model(patchsize=(10,10),clusters=1000)
test_predict(km,clusterModel,"Model_10_1000",patchsize=(10, 10))

km,clusterModel = drive_model(patchsize=(10,10),clusters=2000)
test_predict(km,clusterModel,"Model_10_1000",patchsize=(10, 10))

# Model 4
# Detals : Patch Size = 21 Cluster =50
km,clusterModel = drive_model(patchsize=(21,21),clusters=50)
test_predict(km,clusterModel,"Model_21_50",patchsize=(21,21))

# Model 5
# Detals : Patch Size = 21 Cluster =100
km,clusterModel = drive_model(patchsize=(21,21),clusters=100)
test_predict(km,clusterModel,"Model_21_100",patchsize=(21,21))

# Model 6
# Detals : Patch Size = 21 Cluster =250
km,clusterModel = drive_model(patchsize=(21,21),clusters=250)
test_predict(km,clusterModel,"Model_21_250",patchsize=(21,21))

# Model 7
# Detals : Patch Size = 21 Cluster =500
km,clusterModel = drive_model(patchsize=(21,21),clusters=500)
test_predict(km,clusterModel,"Model_21_500",patchsize=(21,21))

km,clusterModel = drive_model(patchsize=(21,21),clusters=1000)
test_predict(km,clusterModel,"Model_21_1000",patchsize=(21,21))

# Model8
# Detals : Patch Size = 16 Cluster =50
km,clusterModel = drive_model(patchsize=(16,16),clusters=50)
test_predict(km,clusterModel,"Model_16_50",patchsize=(16,16))

# Model9
# Detals : Patch Size = 16 Cluster =100
km,clusterModel = drive_model(patchsize=(16,16),clusters=100)
test_predict(km,clusterModel,"Model_16_100",patchsize=(16,16))

# Model10
# Detals : Patch Size = 16 Cluster =250
km,clusterModel = drive_model(patchsize=(16,16),clusters=250)
test_predict(km,clusterModel,"Model_16_250",patchsize=(16,16))

# Model11
# Detals : Patch Size = 16 Cluster =500
km,clusterModel = drive_model(patchsize=(16,16),clusters=500)
test_predict(km,clusterModel,"Model_16_500",patchsize=(16,16))

km,clusterModel = drive_model(patchsize=(16,16),clusters=1000)
test_predict(km,clusterModel,"Model_16_1000",patchsize=(16,16))

km,clusterModel = drive_model(patchsize=(5,5),clusters=1000)
test_predict(km,clusterModel,"Model_5_1000",patchsize=(5,5))

km,clusterModel = drive_model(patchsize=(7,7),clusters=1000)
test_predict(km,clusterModel,"Model_7_1000",patchsize=(7,7))


# Cluster Model with patch size 15
km,clusterModel = drive_model(patchsize=(15,15),clusters=1000)
test_predict(km,clusterModel,"Model_15_1000",patchsize=(15,15))

mdl = [km,clusterModel]
save_model('../Results/Model_15_1000/model151000.data',mdl)

# Cluster Model with patch size 15
km,clusterModel = drive_model(patchsize=(25,25),clusters=1000)
test_predict(km,clusterModel,"Model_25_1000",patchsize=(25,25))

mdl = [km,clusterModel]
save_model('../Results/Model_25_1000/model251000.data',mdl)

# Cluster Model with patch size 35
km,clusterModel = drive_model(patchsize=(35,35),clusters=1000)
test_predict(km,clusterModel,"Model_35_1000",patchsize=(35,35))

mdl = [km,clusterModel]
save_model('../Results/Model_35_1000/model351000.data',mdl)

# Cluster Model with patch size 50
km,clusterModel = drive_model(patchsize=(50,50),clusters=1000)
test_predict(km,clusterModel,"Model_50_1000",patchsize=(50,50))

mdl = [km,clusterModel]
save_model('../Results/Model_50_1000/model501000.data',mdl)

# Cluster Model with patch size
km,clusterModel = drive_model(patchsize=(100,100),clusters=1000)
km,clusterModel = drive_model(patchsize=(100,100),clusters=1000)

mdl = [km,clusterModel]
save_model('../Results/Model_100_1000/model1001000.data',mdl)


####
#Clahe test_predict
km,clusterModel = drive_model(patchsize=(10,10),clusters=1000)
test_predict(km,clusterModel,"clahe8_10_1000",patchsize=(10,10))

km,clusterModel = drive_model(patchsize=(10,10),clusters=1000)
test_predict(km,clusterModel,"clahe16_10_1000",patchsize=(10,10))

km,clusterModel = drive_model(patchsize=(10,10),clusters=1000)
test_predict(km,clusterModel,"clahe24_10_1000",patchsize=(10,10))


##
#resize image
km,clusterModel = drive_model(patchsize=(10,10),clusters=1000)
test_predict(km,clusterModel,"resize08_10_1000",patchsize=(10,10))

km,clusterModel = drive_model(patchsize=(10,10),clusters=1000)
test_predict(km,clusterModel,"resize06_10_1000",patchsize=(10,10))

km,clusterModel = drive_model(patchsize=(10,10),clusters=1000)
test_predict(km,clusterModel,"resize15_10_1000",patchsize=(10,10))

km,clusterModel = drive_model(patchsize=(21,21),clusters=1000)
test_predict(km,clusterModel,"resize08_21_1000",patchsize=(21,21))

km,clusterModel = drive_model(patchsize=(21,21),clusters=1000)
test_predict(km,clusterModel,"resize06_21_1000",patchsize=(21,21))

km,clusterModel = drive_model(patchsize=(21,21),clusters=1000)
test_predict(km,clusterModel,"resize15_21_1000",patchsize=(21,21))