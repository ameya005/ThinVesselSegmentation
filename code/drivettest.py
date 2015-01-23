'''
One image test
'''
import driveUtils
import patchify

#Load an image
img = plt.imread('../training/images/23_training.tif')

#Seperate R,G,B
imgR = img[:,:,0]
imgG = img[:,:,1]
imgB = img[:,:,2]

#Extract Patches 

patch = patchify.patchify(imgR,(16,16))[0]

'''
Store patches in the working directory
'''

testDict={}
testDict['1']=patch


storePatch(testDict,dataset="testing",nomrmalize="no")
