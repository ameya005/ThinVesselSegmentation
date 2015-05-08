from collections import defaultdict
from skimage.transform import rotate
from sklearn.metrics import precision_recall_curve, auc, roc_curve, accuracy_score, f1_score

__author__ = 'kushal'
"""
Provides utilies and preporcessing facilities
"""

import os
import random
import cPickle as pickle
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import h5py
from skimage import transform
# import cv2


def check_path(path):
    """
    Checks the ending of the given path , and appends '/' to the path
    :type path: string
    """
    if path[-1] != '/':
        path += '/'
    return path


# noinspection PyArgumentList
def compute_random(X, y, samples=10000, seed=42):
    """
    Generated randomly sampled patches for training the classifier

    :rtype : ndarray
    :param X: ndarray
        Array of patches
    :param y: ndarray
        Array of Ground Truth patches
    :param samples: int
        Number of Samples to be extracted per image
    :param seed: int
        Seed for random number generator

    :return:
    """
    x_new = []
    y_new = []
    for key in X.keys():
        rnumber = random.sample(xrange(len(X[key])), samples)
        x_new.extend(X[key][rnumber])
        y_new.extend(y[key][rnumber])
    x_new = np.asarray(x_new)
    y_new = np.asarray(y_new)

    return x_new, y_new


def check_dir_exists(location):
    """
    Check if the directory at given path exists, if not create the directory
    :param location: string
        Path to the directory
    """
    if not os.path.exists(location):
        os.makedirs(location)


def save_image(location, img, name):
    # Correct the location
    """
    Save the image at the given location

    :param location: string
        Path for saving the file
    :param img: ndarray of shape (x,y) or (x,y,z)
        Image to be saved
    :param name: string
        Name of the saved image
    """
    location = check_path(location)

    # Check if directory exists
    check_dir_exists(location)

    # Now save the image
    plt.imsave(location + str(name) + '.png', img)


def save_model(name_model, location, data_save):
    """
    Save the model as h5 file

    :type name_model: object

    """
    save_path = check_path(location) + name_model + '.mdl'

    h5file = h5py.File(save_path, 'w')
    h5file.create_dataset('model', data=data_save)
    h5file.close()


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def read_object(filename):
    with open(filename, 'rb') as output:
        obj = pickle.load(output)
    return obj


def save_model_pickle(name_model, location, data_save):
    """
    Save the model as h5 file

    :type name_model: object

    """
    save_path = check_path(location) + name_model + '.mdl'

    h5file = h5py.File(save_path, 'w')
    h5file.create_dataset('model', data=data_save)
    h5file.close()


def read_model(name, location):
    """
    Read the model
    :return:
    """
    loc = check_path(location) + name
    h5file = h5py.File(loc, 'r')

    mdl = h5file['model'].value
    h5file.close()

    return mdl


def zscore_norm(x, axis=1):
    """
    Normalize the give data

    :param x: ndarray
        The data to be normalized of shape (x,y)
    :param axis: int
        Axis along which to be normalized,
        Defaults to 1, each sample is individually normalized
        For 0, the array is first raveled and then normalized
    :return: ndarray
        Array with zero mean and unit variance
    """
    x = zscore(x, axis=axis)
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = 0

    return x


def preprocessing():
    pass


def rotate_images(imgs, angle=0):
    for i in xrange(imgs.shape[0]):
        imgs[i] = rotate(imgs[i], angle=angle)
    return imgs


def clahe(imgs, tilesize=(8, 8), clplmt=2.0, multi=0):
    clahe_el = cv2.createCLAHE(clipLimit=clplmt, tileGridSize=tilesize)
    imgs = clahe_el.apply(imgs)

    return imgs


# TO DOlist
# TODO: Implement t-sNE for visualization
# TODO: Compute Script

def read_image(path):
    file_list = os.listdir(path)

    img = {os.path.splitext(file)[0][:3]: plt.imread(path + file) for file in file_list}

    return img


def combine_iters(patch_size, clusters, path):
    iters = [9, 10, 15, 21, 33]
    for i, j in enumerate(iters):
        folder_name = 'Drive' + '_p_' + str(j) + '_clus_' + str(clusters)

        if i > 0:
            for key in img.keys():
                img1[key] += img[key]

        img = read_image(check_path(path) + check_path(folder_name))
        if i == 0:
            img1 = img
    # location = 'Drive_p_' + str(patch_size) + '_clus_' + str(clusters)
    location = 'Drive' + '_clus_' + str(clusters)
    check_dir_exists(check_path(path + location))
    for key in img1.keys():
        im = img1[key] / img1[key].max()
        plt.imsave(str(path + location) + '/' + str(key) + '_G' + '.png', im, cmap=plt.cm.gray)


        # for patch_size in [10]:
        # for clusters in [100, 200, 500]:
        # combine_iters(patch_size, clusters, './')


def seg_eval_roc(img, gt):
    """
    Evaluation of segmentation ( FPR,TPR, ROC_AUC)

    Inputs
    ------
    img:	ndarray, Predicted Image
    gt:		ndarray, Ground Truth image

    Returns
    -------
    FPR, TPR , ROC

    """
    img = img.ravel()
    gt = gt.ravel() / 255

    fpr, tpr, _ = roc_curve(gt, img)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def seg_eval_roc(img, gt):
    """
    Evaluation of segmentation ( FPR,TPR, ROC_AUC)

    Inputs
    ------
    img:	ndarray, Predicted Image
    gt:		ndarray, Ground Truth image

    Returns
    -------
    FPR, TPR , ROC

    """
    img = img.ravel()
    gt = gt.ravel() / 255

    fpr, tpr, _ = roc_curve(gt, img)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def seg_eval_prc(img, gt):
    img = img.ravel()
    gt = gt.ravel() / 255.
    precision, recall, thresholds = precision_recall_curve(gt, img)
    roc_auc = auc(precision, recall, reorder=True)
    return precision, recall, thresholds, roc_auc


def plot_roc(fpr, tpr, roc_auc, lkey="Ours"):
    """
    Plot function for ROC curve.
    See : seg_eval_roc() for calculating the given values

    Inputs:
    -------
    FPR,TPR,ROC_AUC
    lkey:	Plot legend value

    """
    # plt.figure()
    plt.plot(fpr, tpr, label=str(lkey) + ' (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def compute_stat(img, gt1_img, mask=None):
    im_pred = []
    im_gt = []
    if mask:
        im_mask = []

    for key in img.keys():
        im_pred.extend((img[key][:, :, 1].ravel()).tolist())
        im_gt.extend((gt1_img[key].ravel()).tolist())
        if mask:
            im_mask.extend((mask[key][:,:,0].ravel()).tolist())

    im_pred = np.asarray(im_pred)
    im_gt = np.asarray(im_gt)
    im_mask = np.asarray(im_mask)

    if mask:
        nonzero = np.nonzero(im_mask)[0]

        im_pred = im_pred[nonzero]
        im_gt = im_gt[nonzero]

    fpr, tpr, roc_auc = seg_eval_roc(im_pred, im_gt)
    precision, recall, thresholds, roc_auc_prc = seg_eval_prc(im_pred, im_gt)
    # acc = accuracy_score(im_gt,im_pred)
    # f1 = f1_score(im_gt, im_pred)
    statistics = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc, 'precision': precision, 'recall': recall,
                  'thresh': thresholds, "auc_prc": roc_auc_prc}
    return statistics, [im_pred, im_gt]

### Write code for statistical analysis ###
stat = defaultdict()
for it in xrange(5):
    loc = 'Drive_iter' + str(it) + '_p10clus1000/'
    img = driveUtils.readimage('./' + loc)

    stat, [imgrav, imggt] = compute_stat(img, gtimages, mask_img)


def find_bestworst(img, gt1_img, mask=None):
    auc_all = defaultdict()
    for key in img.keys():
        im_pred = []
        im_gt = []
        if mask:
            im_mask = []
        im_pred.extend((img[key][:, :, 1].ravel()).tolist())
        im_gt.extend((gt1_img[key].ravel()).tolist())
        if mask:
            im_mask.extend((mask[key].ravel()).tolist())

        im_pred = np.asarray(im_pred)
        im_gt = np.asarray(im_gt)
        im_mask = np.asarray(im_mask)

        if mask:
            nonzero = np.nonzero(im_mask)[0]

            im_pred = im_pred[nonzero]
            im_gt = im_gt[nonzero]

        fpr, tpr, roc_auc = seg_eval_roc(im_pred, im_gt)
        precision, recall, thresholds, roc_auc_prc = seg_eval_prc(im_pred, im_gt)

        statistics = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc, 'precision': precision, 'recall': recall,
                      'thresh': thresholds, "auc_prc": roc_auc_prc}

        auc_all[key] = statistics

    return auc_all


def create_compare(img, gt, thres):
    img = img > thres
    gt /= gt.max()
    comp = np.zeros((img.shape[0], img.shape[1], 3))
    comp[:, :, 0] = (img != gt)
    comp[:, :, 1] = gt != (img == gt)
    comp[:, :, 2] = (img > gt)
    return comp

import pylab as pl

pl.clf()
pl.plot(stat['recall'], stat['precision'], label='Precision-Recall curve')
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.ylim([0.0, 1.05])
pl.xlim([0.0, 1.0])
pl.title('Precision-Recall DRIVE Set: AUC=%0.2f' % stat['auc_prc'])
pl.legend(loc="lower left")
pl.show()