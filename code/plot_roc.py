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

file_green =glob.glob("../Results/5Feb/test/??_G.png")
file_red =glob.glob("../Results/5Feb/test/??_R.png")
file_blue =glob.glob("../Results/5Feb/test/??_B.png")

gt1_img = driveUtils.readimage('../test/1st_manual/')
gt2_img = driveUtils.readimage('../test/2nd_manual/')

img = {}

for file in filenames:
	key = os.path.splitext(file)[0][-2:]
	img[key] = rgb2gray(plt.imread(file)) * mask_img[key]
	img[key] = img[key]/np.max(img[key])

plotroc(img,gt1_img,"All")

for gfile in file_green:
	key = os.path.splitext(gfile)[0][-4:-2]
	img[key] = rgb2gray(plt.imread(gfile)) * mask_img[key]
	img[key] = img[key]/np.max(img[key])


plotroc(img,gt1_img,"Green")

for rfile in file_red:
	key = os.path.splitext(rfile)[0][-4:-2]
	img[key] = rgb2gray(plt.imread(rfile)) * mask_img[key]
	img[key] = img[key]/np.max(img[key])

plotroc(img,gt1_img,"Red")

for bfile in file_blue:
	key = os.path.splitext(bfile)[0][-4:-2]
	img[key] = rgb2gray(plt.imread(bfile)) * mask_img[key]
	img[key] = img[key]/np.max(img[key])

plotroc(img,gt1_img,"Blue")

####################################################################
####                     PLOT OTHERS							####
####################################################################

'''
Read other files
'''
file_CS =glob.glob("../Other_Expt/CS/*.png")
file_DL =glob.glob("../Other_Expt/DL/*.png")
file_SE =glob.glob("../Other_Expt/SE/*.png")
file_RTF =glob.glob("../Other_Expt/RTF/*.png")
file_MICCAI =glob.glob("../Other_Expt/MICCAI/*.png")

for gfile in file_CS:
	key = os.path.splitext(gfile)[0][-2:]
	img[key] = rgb2gray(plt.imread(gfile)) * mask_img[key]
	img[key] = img[key]/np.max(img[key])

plotroc(img,gt1_img,"CS")

for gfile in file_DL:
	key = os.path.splitext(gfile)[0][-2:]
	img[key] = rgb2gray(plt.imread(gfile)) * mask_img[key]
	img[key] = img[key]/np.max(img[key])

plotroc(img,gt1_img,"DL")

for gfile in file_SE:
	key = os.path.splitext(gfile)[0][-2:]
	img[key] = rgb2gray(plt.imread(gfile)) * mask_img[key]
	img[key] = img[key]/np.max(img[key])

plotroc(img,gt1_img,"SE")

for gfile in file_RTF:
	key = os.path.splitext(gfile)[0][-2:]
	img[key] = rgb2gray(plt.imread(gfile)) * mask_img[key]
	img[key] = img[key]/np.max(img[key])

plotroc(img,gt1_img,"RTF")

for gfile in file_MICCAI:
	key = os.path.splitext(gfile)[0][-2:]
	img[key] = rgb2gray(plt.imread(gfile)) * mask_img[key]
	img[key] = img[key]/np.max(img[key])

plotroc(img,gt1_img,"MICCAI")
'''
Calcualte statistics all images
'''
def plotroc(img,gt1_img,tit):

	im_pred = []
	im_gt   = []

	for key in img.keys():
		im_pred.extend((img[key].ravel()).tolist())
		im_gt.extend((gt1_img[key].ravel()).tolist())

	im_pred = np.asarray(im_pred)
	im_gt = np.asarray(im_gt)

	fpr,tpr,roc_auc = driveUtils.seg_eval_roc(im_pred, im_gt)
	driveUtils.plot_roc(fpr, tpr, roc_auc)
	plt.title("ROC_" + str(tit))

#Calculate TP,FP, F1-score, accuracy,precision

acc=[]
prec=[]
reca=[]
f1=[]

for key in th:
    a,b,c,d= driveUtils.seg_eval2(im_pred,im_gt,key)
    acc.append(a)
    prec.append(b)
    reca.append(c)
    f1.append(d)
