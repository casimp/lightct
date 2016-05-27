# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import time
import os

import cv2
from scipy.misc import imsave
import numpy as np
from skimage import color

from lightct.load_scan import LoadProjections


class TomoScan(LoadProjections):
    
    def __init__(self, num_proj, folder, camera_port=0,
                 wait=0, save=True):
        """
        Acquires specified number of projections for analysis and
        reconstruction. The specified number must ensure that a rotational
        range > 360 degrees is captured.

        # num_proj:  Number of projections to acquire (must cover > 360deg)
        # folder:    Path to folder for data storage. Projections and
                     reconstructed slices will be saved here.
        """
        self.folder = folder
        self.p0 = 0
        self.cor_offset = 0
        self.crop = None, None, None
        self.num_images = None
        self.angles = None
        self.recon_data = None

        camera = cv2.VideoCapture(camera_port)
        camera.set(3, 2000)
        camera.set(4, 2000)
        _, im = camera.read()
        try:
            dims = im[:, :, 2].shape + (num_proj, )
        except TypeError:
            error = ('Camera returning None. Check camera settings (port) and'
                     ' ensure camera is not being run by other software.')
            raise TypeError(error)
        self.im_stack = np.zeros(dims)

        print('\n\n')
        # Acquires defined number of images (saves v from hsv)
        for i in range(num_proj):
            _, im = camera.read()
            self.im_stack[:, :, i] = color.rgb2hsv(im)[:, :, 2]
            sys.stdout.write('\rProgress: [{0:20s}] {1:.0f}%'.format('#' *
                             int(20*(i + 1) / num_proj),
                             100*((i + 1)/num_proj)))
            sys.stdout.flush()
            time.sleep(wait)
        del camera

        self.height = self.im_stack.shape[0]
        self.width = self.im_stack.shape[1]

        if save:
            save_folder = os.path.join(self.folder, 'projections')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            for idx in range(self.im_stack.shape[-1]):
                f_path = os.path.join(save_folder, '%04d.tif' % idx)
                imsave(f_path, self.im_stack[:, :, idx])