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
from sklearn.cross_validation import train_test_split

def readimage(dir):
    '''
    Read image changed to match STARE dataset naming
    '''

    files = os.listdir(dir)
    img = {}

    for file in files:
        img[os.path.splitext(file)[0][3:6]] = plt.imread(dir+file)

    return img

def imagesplit():
    '''
    Split the images into training and test set

    '''
    img = readimage('../stare-images/raw/')
    # GT Images
    imgGT = readimage('../stare-images/labels-ah/')

    # Conver the dictionary to Structure array
    imgnd = np.array(img.items(),dtype=dtype)
    imgGTnd = np.array(imgGT.items(),dtype=dtype)

    # Sort the structured arrays so the split key are same
    imgnd = imgnd[imgnd[:,0].argsort()]
    imgGTnd = imgGTnd[imgGTnd[:,0].argsort()]

    #Train Test split
    img_train,img_test,gt_train,gt_test = train_test_split(imgnd,imgGTnd,test_size=0.5,random_state=42)

    #Convert the structured array to dict to reuse old code
    img_train = dict(img_train)
    img_test = dict(img_test)
    gt_train = dict(gt_train)
    gt_test = dict(gt_test)

    return img_train,gt_train,img_test,gt_test

def stare_model(img,imgGT,patchsize=(10,10),clusters=100):
    '''
    Trains the model on STARE dataset

    '''
    #The images are loaded from the imagesplit
    patchesGT = driveUtils.computePatch(imgGT,size=patchsize)
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

def test_predict(test_img,kmG,clusterGtG,location,patchsize=(10,10),rescale=1,clahe=False):
    '''
    Prediction model on the stare dataset

    '''
    #Determine Size of image
    a,b,c = test_img.items()[0][1].shape

    testPatchG = driveUtils.computePatch(test_img, channel=1,size=patchsize)

    if not os.path.exists('../Results/'+str(location)):
        os.makedirs('../Results/'+str(location))    

    test_img_predict ={}

    for key in test_img.keys():     
        tPatchG = driveUtils.zscore_norm(driveUtils.flattenlist(testPatchG[key]))
        # tPatchG = driveUtils.flattenlist(testPatchG[key])
        # print "Debug Level 1.2"
        patchGidx = kmG.predict(tPatchG)
        # print "Debug Level 3"
        testimg = patchify.unpatchify(
            np.asarray([clusterGtG[j] for i, j in enumerate(patchGidx)]), (a,b))

        #testimg = resize(testimg, (584,565))

        test_img_predict[key] = testimg
        print "Saving_" + str(key)

        plt.imsave('../Results/'+str(location)+'/' + str(key) + '_G' +
                   '.png', testimg, cmap=cm.gray)


    return test_img_predict


def stare_call(location):
    '''
    Main call to the STARE Set
    '''
    #Read the images and split the dataset
    train_img,train_gt,test_img,test_gt= imagesplit()
    #Initialize the stare model
    km,clustermodel = stare_model(train_img, train_gt,clusters=1000)
    #Prediction on the test set
    test_img_predict = test_predict(test_img, km, clustermodel,location)

    return test_img_predict,test_gt
