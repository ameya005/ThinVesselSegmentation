#!/usr/bin/env python
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np


__author__ = 'kushal'

from datasets import Drive, Stare, CHASE, ARIA, HRF
from models import KmeansClusterLearn, DictLearn
import utils


def make_all(X, predict_function, **kwargs):
    img = {}
    for key in X.keys():
        img[key] = predict_function(X, **kwargs)

    return img


if __name__ == "__main__":

    def drive_test():
        path_train = '../../Datasets/DRIVE/training'
        path_test = '../../Datasets/DRIVE/test'
        patch_size = (10, 10)
        channel = 1
        ravel = 1
        clusters = 1000
        img_size = (584, 565)
        rotation = 0
        Drive_train = Drive(path_train)

        for patch_size in [(10, 10)]:
            Drive_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
            Drive_train.compute_gt_mask(size=patch_size, mask=1, ravel=1)
            for clusters in [1000]:

                for i in xrange(5):
                    # Extract patches for training

                    if rotation:
                        patch_train, patch_gt_train = utils.compute_random(Drive_train.patches, Drive_train.patchesGT)

                        patch_train = patch_train.reshape(-1, patch_size[0], patch_size[1])
                        patch_gt_train = patch_gt_train.reshape(-1, patch_size[0], patch_size[1])

                        patch_train = np.concatenate((
                            patch_train, utils.rotate_images(patch_train, 30), utils.rotate_images(patch_train, 60),
                            utils.rotate_images(patch_train, 90), utils.rotate_images(patch_train, 120),
                            utils.rotate_images(patch_train, 150)))

                        patch_gt_train = np.concatenate((
                            patch_gt_train, utils.rotate_images(patch_gt_train, 30),
                            utils.rotate_images(patch_gt_train, 60),
                            utils.rotate_images(patch_gt_train, 90), utils.rotate_images(patch_gt_train, 120),
                            utils.rotate_images(patch_gt_train, 150)))

                        patch_train = patch_train.reshape(-1, patch_size[0] * patch_size[1])
                        patch_gt_train = patch_gt_train.reshape(-1, patch_size[0] * patch_size[1])
                    else:
                        patch_train, patch_gt_train = utils.compute_random(Drive_train.patches, Drive_train.patchesGT)
                    # CLuster Model
                    kmmodel = KmeansClusterLearn(n_clusters=clusters, patch_size=patch_size, image_size=img_size,
                                                 normalize=True)
                    kmmodel.fit(patch_train, patch_gt_train)

                    Drive_test = Drive(path_test)

                    Drive_test.compute_patch(size=patch_size, channel=channel, ravel=ravel)
                    # Drive_test.compute_gt_mask(size=patch_size, mask=1, ravel=1)

                    test_img = defaultdict()
                    location = '../Results/Drive_Expt1/' + 'Drive_iter' + str(i) + '_p' + str(
                        patch_size[0]) + 'clus' + str(
                        clusters)
                    utils.check_dir_exists(location)
                    location_model = '../Results/Drive/Models/' + 'Drive_iter_' + str(i) + '_p' + str(
                        patch_size[0]) + 'clus' + str(clusters) + '.mdl'

                    utils.save_object(kmmodel, location_model)
                    for key in Drive_test.patches.keys():
                        test_img[key] = kmmodel.predict_image(Drive_test.patches[key])
                        plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray)


    def dict_model():
        # Dictionary Learning Model
        params = {
            'K': 500,
            'lambda1': 1.0,
            'numThreads': -1,
            'batchsize': 512,
            'iter': 500,
            'posAlpha': True
        }
        cparams = {
            'L': 10,
            'eps': 1.0,
            'numThreads': -1
        }

        dictmodel = DictLearn(n_clusters=clusters, patch_size=patch_size, image_size=img_size, params=params,
                              cparams=cparams)
        dictmodel.fit(patch_train, patch_gt_train)


    # learning different models

    def different_models():
        path_train = '../../Datasets/DRIVE/training'
        path_test = '../../Datasets/DRIVE/test'
        patch_size = (10, 10)
        channel = 1
        ravel = 1
        clusters = 500
        img_size = (584, 565)

    path_model = '../Results/Drive/Models/Drive_iter4_p10clus500.mdl'

    def test_drive(model, dataset):
        kmmodel = utils.read_object(model)

    def drive_test_1():
        path_train = '../../Datasets/DRIVE/training'
        path_test = '../../Datasets/DRIVE/test'
        patch_size = (10, 10)
        channel = 1
        ravel = 1
        clusters = 1000
        img_size = (584, 565)
        rotation = 0
        Drive_train = Drive(path_train)

        Drive_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
        Drive_train.compute_gt_mask(size=patch_size, mask=1, ravel=1)

        if rotation:
            patch_train, patch_gt_train = utils.compute_random(Drive_train.patches, Drive_train.patchesGT)

            patch_train = patch_train.reshape(-1, patch_size[0], patch_size[1])
            patch_gt_train = patch_gt_train.reshape(-1, patch_size[0], patch_size[1])

            patch_train = np.concatenate((
                patch_train, utils.rotate_images(patch_train, 30), utils.rotate_images(patch_train, 60),
                utils.rotate_images(patch_train, 90), utils.rotate_images(patch_train, 120),
                utils.rotate_images(patch_train, 150)))

            patch_gt_train = np.concatenate((
                patch_gt_train, utils.rotate_images(patch_gt_train, 30),
                utils.rotate_images(patch_gt_train, 60),
                utils.rotate_images(patch_gt_train, 90), utils.rotate_images(patch_gt_train, 120),
                utils.rotate_images(patch_gt_train, 150)))

            patch_train = patch_train.reshape(-1, patch_size[0] * patch_size[1])
            patch_gt_train = patch_gt_train.reshape(-1, patch_size[0] * patch_size[1])
        else:
            patch_train, patch_gt_train = utils.compute_random(Drive_train.patches, Drive_train.patchesGT)
        # CLuster Model
        kmmodel = KmeansClusterLearn(n_clusters=clusters, patch_size=patch_size, image_size=img_size,
                                     normalize=True)
        kmmodel.fit(patch_train, patch_gt_train)

        Drive_test = Drive(path_test)

        Drive_test.compute_patch(size=patch_size, channel=channel, ravel=ravel)
        # Drive_test.compute_gt_mask(size=patch_size, mask=1, ravel=1)

        test_img = defaultdict()
        location = '../Results/Drive_Expt1/' + 'Drive_iter' + str(i) + '_p' + str(patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)
        location_model = '../Results/Drive/Models/' + 'Drive_iter_' + str(i) + '_p' + str(
            patch_size[0]) + 'clus' + str(clusters) + '.mdl'

        utils.save_object(kmmodel, location_model)
        for key in Drive_test.patches.keys():
            test_img[key] = kmmodel.predict_image(Drive_test.patches[key])
            plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray)

    def drive_ontrain():
        model_loc = '../Results/Drive/Models/'
        path_train = '../../Datasets/DRIVE/training'
        patch_size = (10, 10)
        channel = 1
        ravel = 1
        clusters = 1000
        img_size = (584, 565)
        rotation = 0
        Drive_train = Drive(path_train)
        Drive_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
        for i in xrange(5):
            mod_name = 'Drive_iter_' + str(i) + '_p' + str(patch_size[0]) + 'clus' + str(clusters) + '.mdl'
            kmmodel = utils.read_object(model_loc + mod_name)

            test_img = defaultdict()
            location = '../Results/Drive_Expt1/train/' + 'Drive_iter' + str(i) + '_p' + str(
                patch_size[0]) + 'clus' + str(
                clusters)
            utils.check_dir_exists(location)

            for key in Drive_train.patches.keys():
                test_img[key] = kmmodel.predict_image(Drive_train.patches[key])
                plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray)

    def drive_onothers():
        model_loc = '../Results/Drive/Models/'
        stare_train = '../../Datasets/STARE/'
        chase_train = '../../Datasets/CHASEDB/'
        aria_train = '../../Datasets/ARIA/'
        drive_train = '../../Datasets/DRIVE/training'
        patch_size = (15, 15)
        channel = 1
        ravel = 1
        clusters = 1000
        img_size = (584, 565)
        rotation = 0

        i = 1
        mod_name = 'Drive_iter' + str(i) + '_p' + str(patch_size[0]) + 'clus' + str(clusters) + '.mdl'
        kmmodel = utils.read_object(model_loc + mod_name)

        print "Start on Drive Train"
        # Test on Stare
        # img_size = (605, 700)
        # kmmodel.image_size = img_size
        dataset_train = Drive(drive_train)
        dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)

        test_img = defaultdict()
        location = '../Results/Drive_Expt2/train/drive/' + 'Drive_iter' + str(i) + '_p' + str(
            patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)

        for key in dataset_train.patches.keys():
            test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
            plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')

        print "Start on stare"
        # Test on Stare
        img_size = (605, 700)
        kmmodel.image_size = img_size
        dataset_train = Stare(stare_train)
        dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)

        test_img = defaultdict()
        location = '../Results/Drive_Expt2/train/stare/' + 'Drive_iter' + str(i) + '_p' + str(
            patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)

        for key in dataset_train.patches.keys():
            test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
            plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')
        print "start on chase"
        # Test on chase
        dataset_train = CHASE(chase_train)
        dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
        img_size = (960, 999)
        kmmodel.image_size = img_size
        test_img = defaultdict()
        location = '../Results/Drive_Expt2/train/chase/' + 'Drive_iter' + str(i) + '_p' + str(
            patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)

        for key in dataset_train.patches.keys():
            test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
            plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')

        print "start on aria"
        # Test on chase
        dataset_train = ARIA(aria_train)
        dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
        img_size = (576, 768)
        kmmodel.image_size = img_size
        test_img = defaultdict()
        location = '../Results/Drive_Expt2/train/aria/' + 'Drive_iter' + str(i) + '_p' + str(
            patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)

        for key in dataset_train.patches.keys():
            test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
            plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')

    def chase_model():

        model_loc = '../Results/Drive/Models/'
        stare_train = '../../Datasets/STARE/'
        chase_train = '../../Datasets/CHASEDB/'
        aria_train = '../../Datasets/ARIA/'
        drive_train = '../../Datasets/DRIVE/training'
        patch_size = (15, 15)
        channel = 1
        ravel = 1
        clusters = 1000
        img_size = (584, 565)
        rotation = 0
        i = 1
        # kmmodel = utils.read_object(model_loc + mod_name)
        # Train on chase DB
        dataset_train = CHASE(chase_train)
        dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
        dataset_train.compute_gt_mask(size=patch_size, ravel=ravel)
        img_size = (960, 999)

        patch_train, patch_gt_train = utils.compute_random(dataset_train.patches, dataset_train.patchesGT)
        # CLuster Model
        kmmodel = KmeansClusterLearn(n_clusters=clusters, patch_size=patch_size, image_size=img_size,
                                     normalize=True)
        kmmodel.fit(patch_train, patch_gt_train)

        location = '../Results/Chase/Chase_Expt1/' + 'Chase_iter' + str(i) + '_p' + str(patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)
        # location_model = '../Results/Chase/Models/' + 'Chase_iter_' + str(i) + '_p' + str(
        # patch_size[0]) + 'clus' + str(clusters) + '.mdl'
        #
        # # utils.save_object(kmmodel, location_model)

        # Prediction on others
        stare_train = '../../Datasets/STARE/'
        chase_train = '../../Datasets/CHASEDB/'
        aria_train = '../../Datasets/ARIA/'
        drive_train = '../../Datasets/DRIVE/training/'
        drive_test = '../../Datasets/DRIVE/test/'

        print "Start on Drive Train"
        # Test on Stare
        img_size = (584, 565)
        kmmodel.image_size = img_size
        dataset_train = Drive(drive_train)
        dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)

        test_img = defaultdict()
        location = '../Results/Chase/' + 'drivetrain_iter' + str(i) + '_p' + str(
            patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)

        for key in dataset_train.patches.keys():
            test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
            plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')

        print "Start on Drive Test"
        # Test on Stare
        img_size = (584, 565)
        kmmodel.image_size = img_size
        dataset_train = Drive(drive_test)
        dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)

        test_img = defaultdict()
        location = '../Results/Chase/' + 'drivetest_iter' + str(i) + '_p' + str(
            patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)

        for key in dataset_train.patches.keys():
            test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
            plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')

        print "Start on stare"
        # Test on Stare
        img_size = (605, 700)
        kmmodel.image_size = img_size
        dataset_train = Stare(stare_train)
        dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)

        test_img = defaultdict()
        location = '../Results/Chase/' + 'stare_iter' + str(i) + '_p' + str(
            patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)

        for key in dataset_train.patches.keys():
            test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
            plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')
        print "start on chase"
        # Test on chase
        dataset_train = CHASE(chase_train)
        dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
        img_size = (960, 999)
        kmmodel.image_size = img_size
        test_img = defaultdict()
        location = '../Results/Chase/' + 'Chase_iter' + str(i) + '_p' + str(
            patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)

        for key in dataset_train.patches.keys():
            test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
            plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')

        print "start on aria"
        # Test on chase
        dataset_train = ARIA(aria_train)
        dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
        img_size = (576, 768)
        kmmodel.image_size = img_size
        test_img = defaultdict()
        location = '../Results/Chase/' + 'aria_iter' + str(i) + '_p' + str(
            patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)

        for key in dataset_train.patches.keys():
            test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
            plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')

    def stare_model():

        model_loc = '../Results/Drive/Models/'
        stare_train = '../../Datasets/STARE/'
        chase_train = '../../Datasets/CHASEDB/'
        aria_train = '../../Datasets/ARIA/'
        drive_train = '../../Datasets/DRIVE/training'
        patch_size = (15, 15)
        channel = 1
        ravel = 1
        clusters = 1000
        img_size = (584, 565)
        rotation = 0
        i = 1
        # kmmodel = utils.read_object(model_loc + mod_name)
        # Train on chase DB
        dataset_train = Stare(stare_train)
        dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
        dataset_train.compute_gt_mask(size=patch_size, ravel=ravel)
        img_size = (605, 700)

        patch_train, patch_gt_train = utils.compute_random(dataset_train.patches, dataset_train.patchesGT)
        # CLuster Model
        kmmodel = KmeansClusterLearn(n_clusters=clusters, patch_size=patch_size, image_size=img_size,
                                     normalize=True)
        kmmodel.fit(patch_train, patch_gt_train)

        location = '../Results/stare/stare/' + 'stare_iter' + str(i) + '_p' + str(patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)
        # # location_model = '../Results/stare/Models/' + 'stare_iter_' + str(i) + '_p' + str(
        # patch_size[0]) + 'clus' + str(clusters) + '.mdl'

        # utils.save_object(kmmodel, location_model)

        # Prediction on others
        stare_train = '../../Datasets/STARE/'
        chase_train = '../../Datasets/CHASEDB/'
        aria_train = '../../Datasets/ARIA/'
        drive_train = '../../Datasets/DRIVE/training/'
        drive_test = '../../Datasets/DRIVE/test/'

        print "Start on Drive Train"
        # Test on Stare
        img_size = (584, 565)
        kmmodel.image_size = img_size
        dataset_train = Drive(drive_train)
        dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)

        test_img = defaultdict()
        location = '../Results/stare/' + 'drivetrain_iter' + str(i) + '_p' + str(
            patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)

        for key in dataset_train.patches.keys():
            test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
        plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')

        print "Start on Drive Test"
        # Test on Stare
        img_size = (584, 565)
        kmmodel.image_size = img_size
        dataset_train = Drive(drive_test)
        dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)

        test_img = defaultdict()
        location = '../Results/stare/' + 'drivetest_iter' + str(i) + '_p' + str(
            patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)

        for key in dataset_train.patches.keys():
            test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
        plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')

        print "Start on stare"
        # Test on Stare
        img_size = (605, 700)
        kmmodel.image_size = img_size
        dataset_train = Stare(stare_train)
        dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)

        test_img = defaultdict()
        location = '../Results/stare/' + 'stare_iter' + str(i) + '_p' + str(
            patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)

        for key in dataset_train.patches.keys():
            test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
        plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')
        print "start on chase"
        # Test on chase
        dataset_train = CHASE(chase_train)
        dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
        img_size = (960, 999)
        kmmodel.image_size = img_size
        test_img = defaultdict()
        location = '../Results/stare/' + 'Chase_iter' + str(i) + '_p' + str(
            patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)

        for key in dataset_train.patches.keys():
            test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
        plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')

        print "start on aria"
        # Test on chase
        dataset_train = ARIA(aria_train)
        dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
        img_size = (576, 768)
        kmmodel.image_size = img_size
        test_img = defaultdict()
        location = '../Results/stare/' + 'aria_iter' + str(i) + '_p' + str(
            patch_size[0]) + 'clus' + str(
            clusters)
        utils.check_dir_exists(location)

        for key in dataset_train.patches.keys():
            test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
        plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')


def aria_model():
    model_loc = '../Results/Drive/Models/'
    stare_train = '../../Datasets/STARE/'
    chase_train = '../../Datasets/CHASEDB/'
    aria_train = '../../Datasets/ARIA/'
    drive_train = '../../Datasets/DRIVE/training'
    patch_size = (15, 15)
    channel = 1
    ravel = 1
    clusters = 1000
    img_size = (584, 565)
    rotation = 0
    i = 1
    # kmmodel = utils.read_object(model_loc + mod_name)
    # Train on chase DB
    dataset_train = ARIA(aria_train)
    dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
    dataset_train.compute_gt_mask(size=patch_size, ravel=ravel)
    img_size = (576, 768)

    patch_train, patch_gt_train = utils.compute_random(dataset_train.patches, dataset_train.patchesGT)
    # CLuster Model
    kmmodel = KmeansClusterLearn(n_clusters=clusters, patch_size=patch_size, image_size=img_size,
                                 normalize=True)
    kmmodel.fit(patch_train, patch_gt_train)

    location = '../Results/aria/aria/' + 'aria_iter' + str(i) + '_p' + str(patch_size[0]) + 'clus' + str(
        clusters)
    utils.check_dir_exists(location)
    # location_model = '../Results/aria/Models/' + 'stare_iter_' + str(i) + '_p' + str(
    #     patch_size[0]) + 'clus' + str(clusters) + '.mdl'
    #
    # utils.save_object(kmmodel, location_model)

    # Prediction on others
    stare_train = '../../Datasets/STARE/'
    chase_train = '../../Datasets/CHASEDB/'
    aria_train = '../../Datasets/ARIA/'
    drive_train = '../../Datasets/DRIVE/training/'
    drive_test = '../../Datasets/DRIVE/test/'

    print "Start on Drive Train"
    # Test on Stare
    img_size = (584, 565)
    kmmodel.image_size = img_size
    dataset_train = Drive(drive_train)
    dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)

    test_img = defaultdict()
    location = '../Results/aria/' + 'drivetrain_iter' + str(i) + '_p' + str(
        patch_size[0]) + 'clus' + str(
        clusters)
    utils.check_dir_exists(location)

    for key in dataset_train.patches.keys():
        test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
        plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')

    print "Start on Drive Test"
    # Test on Stare
    img_size = (584, 565)
    kmmodel.image_size = img_size
    dataset_train = Drive(drive_test)
    dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)

    test_img = defaultdict()
    location = '../Results/aria/' + 'drivetest_iter' + str(i) + '_p' + str(
        patch_size[0]) + 'clus' + str(
        clusters)
    utils.check_dir_exists(location)

    for key in dataset_train.patches.keys():
        test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
        plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')

    print "Start on stare"
    # Test on Stare
    img_size = (605, 700)
    kmmodel.image_size = img_size
    dataset_train = Stare(stare_train)
    dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)

    test_img = defaultdict()
    location = '../Results/aria/' + 'stare_iter' + str(i) + '_p' + str(
        patch_size[0]) + 'clus' + str(
        clusters)
    utils.check_dir_exists(location)

    for key in dataset_train.patches.keys():
        test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
        plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')
    print "start on chase"
    # Test on chase
    dataset_train = CHASE(chase_train)
    dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
    img_size = (960, 999)
    kmmodel.image_size = img_size
    test_img = defaultdict()
    location = '../Results/aria/' + 'Chase_iter' + str(i) + '_p' + str(
        patch_size[0]) + 'clus' + str(
        clusters)
    utils.check_dir_exists(location)

    for key in dataset_train.patches.keys():
        test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
        plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')

    print "start on aria"
    # Test on chase
    dataset_train = ARIA(aria_train)
    dataset_train.compute_patch(size=patch_size, channel=channel, ravel=ravel)
    img_size = (576, 768)
    kmmodel.image_size = img_size
    test_img = defaultdict()
    location = '../Results/aria/' + 'aria_iter' + str(i) + '_p' + str(
        patch_size[0]) + 'clus' + str(
        clusters)
    utils.check_dir_exists(location)

    for key in dataset_train.patches.keys():
        test_img[key] = kmmodel.predict_image(dataset_train.patches[key])
        plt.imsave(str(location) + '/' + str(key) + '_G' + '.png', test_img[key], cmap=plt.cm.gray, format='png')



