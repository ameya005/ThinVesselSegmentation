import driveUtils
import os
import glob
from skimage.color import rgb2gray
import numpy as np

'''
Read the files
'''

filenames = glob.glob("../Results/5Feb/test/??.png")
mask_img = driveUtils.readimage('../test/mask/')

gt1_img = driveUtils.readimage('../test/1st_manual/')
gt2_img = driveUtils.readimage('../test/2nd_manual/')

img = {}

for file in filenames:
	key = os.path.splitext(file)[0][-2:]
	img[key] = rgb2gray(plt.imread(file)) * mask_img[key]
	img[key] = img[key]/np.max(img[key])


'''
Calcualte statistics all images
'''
im_true = []
im_gt   = []

for key in img.keys():
	im_true.extend((img[key].ravel()).tolist())
	im_gt.extend((gt1_img[key].ravel()).tolist())

im_true = np.asarray(im_true)
im_gt = np.asarray(im_gt)
