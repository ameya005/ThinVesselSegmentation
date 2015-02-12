import driveUtils
import os
import glob
from skimage.color import rgb2gray
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

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
im_pred = []
im_gt   = []

for key in img.keys():
	im_pred.extend((img[key].ravel()).tolist())
	im_gt.extend((gt1_img[key].ravel()).tolist())

im_pred = np.asarray(im_pred)
im_gt = np.asarray(im_gt)

fpr,tpr,roc_auc = driveUtils.seg_eval_roc(im_pred, im_gt)
driveUtils.plot_roc(fpr, tpr, roc_auc)

#Calculate TP,FP, F1-score, accuracy,precision

