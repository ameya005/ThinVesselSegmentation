'''
Dictionary Learning
'''
import os
import glob
from matplotlib import pyplot as plt
import patchify

import numpy as np
from scipy.stats import zscore
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from skimage.morphology import binary_erosion, disk,rectangle,binary_closing
import cPickle as pickle
from skimage import exposure
import cv2
import skimage.transform as skt
from sklearn.decomposition import MiniBatchDictionaryLearning,DictionaryLearning
import random
import driveUtils
from sklearn.decomposition import SparseCoder

def creatGTdict(alpha,gtPatches):
    ''' Creates GT Dict

    Read the code from the training set and creates a dictionary mapping the Ground truth

    '''
    sA,sB = alpha.shape

    posDict = {}
    negDict = {}

    for i in range(sA):
        row = alpha.getrow(i)
        row = row.toarray()

        pos = np.where(row>0)[1]
        neg = np.where(row<0)[1]

        # posIndx = [row[0][x]*gtPatches[x] for x in pos]
        # negIndx = [row[0][x]*gtPatches[x] for x in neg]

        posIndx = [gtPatches[x] for x in pos]
        negIndx = [gtPatches[x] for x in neg]
        
        posWt = [row[0][x] for x in pos]
        negWt = [row[0][x] for x in neg]

        posDict[i] = np.average(posIndx,axis=0,weights=posWt)
        negDict[i] = np.average(negIndx,axis=0,weights=negWt)

    return posDict,negDict

def dictlearn(patchsize=(10,10),clusters=500,clahe=False,rescale=1,thres = 0, spams=False):

	## Extracting patches
    img = driveUtils.readimage('../training/images/')
    #img = driveUtils.dictimgscale(img,scaling=rescale) #Scaling images

    # Segmentation Patches
    imgGT = driveUtils.readimage('../training/1st_manual/')
    #imgGT = driveUtils.dictimgscale(imgGT,scaling=rescale) # Scaling

    patchesGT = driveUtils.computePatch(imgGT,size=patchsize)
    # Generate Random numbexs

    patchesGreen = driveUtils.computePatch(img, channel=1,size=patchsize)

    greenPatch = []
    greenPatchGT = []

    # Extracting Ranom patches
    for key in patchesGreen.keys():
        # rnumber = (np.random.sample(15000) * 250000).astype('int')
        rnumber = random.sample(xrange(len(patchesGreen[key])), 10000)
        rnumber.extend(arange(100))
        greenPatch.extend(patchesGreen[key][rnumber])
        greenPatchGT.extend(patchesGT[key][rnumber])

    greenPatch = driveUtils.flattenlist(greenPatch)
    greenPatch = driveUtils.zscore_norm(greenPatch)
    # greenPatchGT = driveUtils.flattenlist(greenPatchGT)

    ##########################################################################################
    '''
    If using SPAMS
    '''
    if spams:

        try:
            import spams
        except ImportError:
            print "Please install SPAMS library"

        X = np.asarray(greenPatch)
        X = np.asfortranarray(X.T)

        # Parameters for the trainDL from SPAMS library
        param = {
        'K' : 500,
        'lambda1' : 0.01,
        'numThreads' : -1,
        'batchsize' : 400,
        'iter' : 1000,
        'posAlpha': True
        }

        # Train the model to obtain the dictionary
        D = spams.trainDL(X,**param)
        # Predicting

        L = 10
        eps = 1.0
        numThreads = -1
        
        greenIdx = spams.omp(X,D,L=L,eps= eps,return_reg_path = False,numThreads = numThreads)
        # Making the patches scale to [0,1]
        posD,negD =creatGTdict(greenIdx, np.asarray(greenPatchGT)/255)

        return D,posD,negD

    if not spams:
         # Zscore Normaliztion
   
       	#Ground truth 
        
        #greenPatchGT = driveUtils.zscore_norm(greenPatchGT)
        ##################################################################
        # Learn Dictionary

        #km = MiniBatchDictionaryLearning(n_jobs=-1,n_components=500,fit_algorithm='cd',transform_algorithm='lasso_cd',verbose=2,random_state=42)
        km = MiniBatchDictionaryLearning(n_jobs=-1,n_components=clusters,fit_algorithm='cd',transform_algorithm='threshold',verbose=2,random_state=42,transform_alpha=0.2,batch_size=10)

        km.fit(greenPatch)

        #################################################################
        greenIdx = km.transform(greenPatch)

        # Work around
        
        idx = (greenIdx > thres).astype(int)
        
        # sum of the idx ( gives number of elements used to construct the dictGt)
        idxSum = idx.T.sum(axis=1)
        idxSum = idxSum.reshape(idxSum.shape[0],1).astype(int)


        groundPatch = np.array(greenPatchGT,dtype='int')

        dictComponents = km.components_
        
        # Dict Ground is obtained here

        dictGround = np.dot(idx.T,groundPatch)
        # Normalize the dictGround 
        groundDict = dictGround / idxSum

        return km,groundDict

def test_predict(km,groundDict,location,patchsize=(10,10),rescale=1,clahe=False,thres=0):

    test_img = driveUtils.readimage('../test/images/')
    #test_img = driveUtils.dictimgscale(test_img,scaling=1) #Scaling
    

    
    #Determine Size of image
    a,b,c = test_img['11'].shape

    testPatchG = driveUtils.computePatch(test_img, channel=1,size=patchsize)

    if not os.path.exists('../Results/'+str(location)):
        os.makedirs('../Results/'+str(location))

    for key in test_img.keys():     
        tPatchG = driveUtils.zscore_norm(driveUtils.flattenlist(testPatchG[key]))
        # tPatchG = driveUtils.flattenlist(testPatchG[key])
        # print "Debug Level 1.2"
        patchGidx = (km.transform(tPatchG))
        patchGidx[ (patchGidx >thres) ] = 1
        patchGidx[patchGidx <=thres] =0
        patchSum = (patchGidx.sum(axis=1))
        patchSum = patchSum.reshape(patchSum.shape[0],1)
        # print "Debug Level 3"
        testimg = np.dot(patchGidx,groundDict)
        testimg = testimg / patchSum
        testimg = testimg.reshape(testimg.shape[0],patchsize[0],patchsize[1])

        img = patchify.unpatchify(testimg,(a,b))

        #testimg = resize(testimg, (584,565))

        print "Saving_" + str(key)

        plt.imsave('../Results/'+str(location)+'/' + str(key) + '_G' +
                   '.png', img, cmap=cm.gray)

# with open('dictModel1', 'rb') as fp:
#     a=pickle.load(fp)

# Sparse Coder
coder = SparseCoder(dictionary=D,transform_n_nonzero_coefs=10,transform_alpha=15,transform_algorithm='omp')

# Test for SPAMS Coder

def spams_test(D,posD,negD,param,key='11'):

    test_img = driveUtils.readimage('../test/images/')
    # Dimesnions of the read image
    a,b,c = test_img['11'].shape
    # Patch the image Channel : Green
    testPatchG = driveUtils.computePatch(test_img, channel=1,size=patchsize)
    # Compute the patches
    tPatchG = np.asarray(driveUtils.zscore_norm(driveUtils.flattenlist(testPatchG[key])))

    data = np.asfortranarray(tPatchG.T)
    
    # Ground Dict
    # dG = ( np.asarray(posD.values()) + np.asarray(negD.values()) ) / 2
    # dG = dG.reshape(dG.shape[0],-1)
    # # Parameters for OMP

    # Ground Dict Postive and negative
    dGP = np.asarray(posD.values()) 
    dGP = dGP.reshape(dGP.shape[0],-1)
    dGN = np.asarray(negD.values()) 
    dGN = dGN.reshape(dGN.shape[0],-1)

    L = 50
    eps = 1.0
    numThreads = -1
    
    # Compute the sparse code
    code = spams.omp(data,D,L=L,eps= eps,return_reg_path = False,numThreads = numThreads)

    # FIXME
    code = code.toarray()
    code = code.T

    codeP = (code>0).astype(int)
    codeN = (code<=0).astype(int)

    # timg = np.dot(codeP,dG)
    # timg = timg.reshape(timg.shape[0],10,10)
    # pimg = patchify.unpatchify(timg,(a,b))

    timgP = np.dot(codeP,dGP)
    timgP = timgP.reshape(timgP.shape[0],10,10)
    pimg = patchify.unpatchify(timgP,(a,b))

    timgN = np.dot(codeN,dGN)
    timgN = timgN.reshape(timgN.shape[0],10,10)
    nimg = patchify.unpatchify(timgN,(a,b))