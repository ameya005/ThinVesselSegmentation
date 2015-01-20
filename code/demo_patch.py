
'''
Given an Image generate patches of given size for all the pixels in the image.
The example in this program computes the patches of an image and restich those patches to 
generate the orignal image

'''
import patchify
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
import numpy as np

img = rgb2gray(imread('train.tif'))

patches,size = patchify.patchify(img,(40,40))

newimg = patchify.unpatchify(patches,size)

plt.figure(1),plt.imshow(patches[1])
plt.figure(2),plt.imshow(newimg)

'''
We replace some patches in the image with random patches and restitch the image
Create random patches

'''
x1 = np.random.rand(40,40)*256
x2 = np.random.rand(40,40)*256

# Replace patches

patches[1000]=x1
patches[30000]=x2


rimage = patchify.unpatchify(patches, size)

plt.figure('3'),plt.imshow(rimage)