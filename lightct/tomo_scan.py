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

def image_acquisition(num_proj, camera_port=0, wait=0, hsv='v'):
    hsv_dict = {'h':0, 's':1, 'v':2}
    camera = cv2.VideoCapture(camera_port)
    camera.set(3, 2000)
    camera.set(4, 2000)
    try:
        dims = camera.read()[1][:, :, 2].shape + (num_proj, )
    except TypeError:
        error = ('Camera returning None. Check camera settings (port) and'
                 ' ensure camera is not being run by other software.')
        raise TypeError(error)
    im_stack = np.zeros(dims)

    # Acquires defined number of images (saves slice from hsv)
    for i in range(num_proj):
        _, im = camera.read()
        im_stack[:, :, i] = color.rgb2hsv(im)[:, :, hsv_dict[hsv]]
        sys.stdout.write('\rProgress: [{0:20s}] {1:.0f}%'.format('#' *
                         int(20*(i + 1) / num_proj),
                         100*((i + 1)/num_proj)))
        sys.stdout.flush()
        time.sleep(wait)
    del camera
    return im_stack


class TomoScan(LoadProjections):
    
    def __init__(self, num_proj, folder, camera_port=0,
                 wait=0, save=True, hsv='v'):
        """
        Acquires specified number of projections for analysis and
        reconstruction. The specified number must ensure that a rotational
        range > 360 degrees is captured.

        # num_proj:  Number of projections to acquire (must cover > 360deg)
        # folder:    Path to folder for data storage. Projections and
                     reconstructed slices will be saved here.
        # hsv:       Extract either the hue (h), saturation (s) or variance(v)
                     from the image matrix.
        """
        self.folder = folder
        self.p0 = 0
        self.cor_offset = 0
        self.crop = None, None, None, None
        self.num_images = None
        self.angles = None
        self.recon_data = None
        
        self.im_stack = image_acquisition(num_proj, camera_port, wait, hsv)
        self.height, self.width = self.im_stack.shape[:2]

        if save:
            save_folder = os.path.join(self.folder, 'projections')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            for idx in range(self.im_stack.shape[-1]):
                f_path = os.path.join(save_folder, '%04d.tif' % idx)
                imsave(f_path, self.im_stack[:, :, idx])