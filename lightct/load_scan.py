# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imsave
from scipy.signal import medfilt, argrelmin
from skimage.transform import iradon, iradon_sart, downscale_local_mean

from lightct.plot_funcs import recentre_plot, SetAngleInteract


class LoadProjections(object):

    def __init__(self, folder, im_type='tif'):
        """
        Load a previously acquired series of projections for analysis and
        reconstruction.

        # folder:    Path to folder where projections are stored. Reconstructed
                     slices will also be saved here.
        """
        self.folder = folder
        self.p0 = 0
        self.cor_offset = 0
        self.crop = None, None, None, None
        self.num_images = None
        self.angles = None

        files = [f for f in os.listdir(folder) if f[-4:] == '.%s' % im_type]
        im_shape = imread(os.path.join(self.folder, files[0])).shape
        self.im_stack = np.zeros(im_shape + (len(files), ))
        for idx, fname in enumerate(files):
                sys.stdout.write('\rProgress: [{0:20s}] {1:.0f}%'.format('#' *
                                 int(20*(idx + 1) / len(files)),
                                 100*((idx + 1)/len(files))))
                sys.stdout.flush()
                f = os.path.join(self.folder, fname)
                self.im_stack[:, :, idx] = imread(f)

        self.height = self.im_stack.shape[0]
        self.width = self.im_stack.shape[1]
        
    def plot_histogram(self, proj=5):
        """
        Plots histogram of pixel intensity for specified projection.
        
        # proj:       Number of projection to display/assess (int)
        """
        histogram = np.histogram(self.im_stack[:, :, proj], 255)
        plt.plot(histogram[0])
        plt.show()

    def set_angles(self, num_images, ang_range=360, p0=5):
        """
        Manually define the number of images in 360 degrees. 

        # num_images: Specify number of images
        # ang_range:  Specify angular range (must be multiple of 180)
        # p0:         Projection to use as initial or reference projection.
                      Recommended to be greater than 1 (due to acquisition
                      spacing issues in initial projections)

        """
        self.p0 = p0

        error = 'Images must cover a rotational range of 180 or 360 deg'
        assert (ang_range == 180) or (ang_range == 360), error
        self.angles = np.linspace(0, ang_range, num_images)
        self.num_images = num_images
        
    def auto_set_angles(self, est_nproj, p0=5, downscale=True, plot=True):
        """
        Attempts to automatically locate image at 360 degrees (and multiples
        of 360 degrees). Alignment based on difference calculation between 
        reference projection each subsequent projections. 
        
        # est_nproj:  Estimated number of projections in 360 degrees
        # p0:         Projection to use as initial or reference projection.
                      Recommended to be greater than 1 (due to acquisition
                      spacing issues in initial projections)
        # plot:       Plot the difference results
        """
        order = est_nproj // 2
        self.p0 = p0
        
        diff = np.nan * np.ones((self.im_stack.shape[-1] - p0))
        proj_nums = range(p0, self.im_stack.shape[-1])

        ref = self.im_stack[:, :, p0]
        data = self.im_stack[:, :, p0:]
        
        tmp = data - ref[:, :, np.newaxis]
        tmp = tmp.reshape(-1, tmp.shape[-1])
    
        diff = tmp.std(axis=0)

        minimas = argrelmin(diff, order=order)
        self.num_images = minimas[0][0] + 1
        self.angles = np.linspace(0, 360, self.num_images, dtype=int)
        
        if plot:
            plt.figure()
            plt.plot(proj_nums, diff)
            plt.plot(minimas[0] + p0, np.array(diff)[minimas], 'r*')
            plt.plot([minimas[0][0] + p0, minimas[0][0] + p0],
                     [0, np.max(diff)], 'r--')
            plt.xlabel('Image number')
            plt.ylabel('Thresholded Pixels Relative to Image 1')
            plt.text(minimas[0][0] + p0, np.max(diff), r'$360^{\circ}$',
                     horizontalalignment='center', verticalalignment='bottom')
        plt.show()
                     
        print('\n%i images in a 360 rotation.\n\nIf this is incorrect '
              'either rerun with a different value for est_nproj or use the '
              'manual method.' % self.num_images)
        
    def set_centre(self, cor):
        """
        Define the centre of rotation.
        
        # window:     Window width to search across (pixels).
        """
        self.cor_offset = cor
        
    def auto_centre(self, window=400, downsample_y=2, plot=True):
        """
        Automatic method for finding the centre of rotation.
        
        # window:     Window width to search across (pixels).
        """
        downsample = (downsample_y, 1)
        half_win = window // 2
        win_range = range(-half_win, half_win)
        
        # Compare ref image with flipped 180deg counterpart
        ref_image = self.im_stack[:, half_win:-half_win, self.p0]
        ref = downscale_local_mean(ref_image, downsample)
        im_180_image = self.im_stack[:, :, self.num_images // 2 + self.p0]
        im_180 = downscale_local_mean(im_180_image, downsample)
        flipped = np.fliplr(im_180)
        
        diff = np.nan * np.zeros(len(win_range))

        # Flip win_range as we are working on flipped data
        for idx, i in enumerate(win_range[::-1]):
            
            cropped = flipped[:, half_win + i: -half_win + i]
            tmp = cropped - ref
            diff[idx] = tmp.std()
            
        minima = np.argmin(diff)
        self.cor_offset = win_range[minima]
        print('COR = %i' % self.cor_offset)

        if plot:
            plt.plot([i for i in win_range], diff)
            plt.plot(self.cor_offset, np.min(diff), '*')
            plt.ylabel('Standard deviation (original v 180deg flipped)')
            plt.xlabel('2 * Centre correction (pixels)')
            im_copy = np.copy(self.im_stack[:, :, self.p0])
            recentre_plot(im_copy, self.cor_offset)
        
    def manual_set_angles(self, p0=5):
        """
        Manually define the number of images in 360 degrees. Defaults to 
        interactive mode in which images can be compared against initial, 
        reference image.

        # p0:         Projection to use as initial or reference projection.
                      Recommended to be greater than 1 (due to acquisition
                      spacing issues in initial projections)
        """
        self.p0 = p0

        interact = SetAngleInteract(self.im_stack, self.p0)
        interact.interact()
        self.angles = interact.angles
        self.num_images = interact.num_images
        
    def set_crop(self, width, top, bottom, plot=True):
        """
        Crop...
        """
        if self.cor_offset >= 0:
            images = self.im_stack[:, self.cor_offset:]
        else:
            images = self.im_stack[:, :self.cor_offset]
            
        self.crop = ()
        for i in (width, -width, top, -bottom): 
            self.crop += (None,) if i == 0 else (i,)

        
        if plot:
            left, right, top, bottom = self.crop
            
            images_per_degree = self.num_images / 360
            degrees = [0, 60, 120]
            image_nums = [int(images_per_degree * deg) for deg in degrees]
            fig, ax_array = plt.subplots(1, 3, figsize=(8, 3))
            for ax, i in zip(ax_array, image_nums):
                ax.imshow(images[top:bottom, left:right, i])
                ax.text(0.15, 0.88, r'$%0d^\circ$' % (i * 360/self.num_images), 
                        size=14, transform = ax.transAxes,
                        va = 'center', ha = 'center')
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])
            fig.tight_layout()
            plt.show()
                
    def reconstruct(self, downsample=(4, 4), crop=True, median_filter=False, 
                    kernel=9, recon_alg='fbp', sart_iters=1, 
                    crop_circle=True, save=True):
        """
        Reconstruct the data using a radon transform. Reconstructed slices
        saved in folder specified upon class creation.

        # downsample: Downsample (local mean) data before reconstructing.
                      Specify mean kernel size (height, width).
        # pre_filter: If True apply median filter to data before reconstructing
        # kernel:     Kernel size to use for median filter
        """
            
        if self.cor_offset >= 0:
            images = self.im_stack[:, self.cor_offset:]
        else:
            images = self.im_stack[:, :self.cor_offset]
            
        images = images[:, :, self.p0:self.num_images + self.p0]
        
        if crop:
            left, right, top, bottom = self.crop
            images = images[top:bottom, left:right]
            
        images = downscale_local_mean(images, downsample + (1, ))
        recon_height, recon_width = images.shape[:2]
        self.recon_data = np.zeros((recon_width, recon_width, recon_height))
        
#        if crop_circle:
#            w = int((recon_width**2 / 2)**0.5) 
#            w = w if (w - recon_width) % 2 == 0 else w - 1
#            w0 =   int((recon_width - w) / 2  )
#            wf = int(w0 + w)
#            self.recon_data = self.recon_data[w0:wf, w0:wf]

        if median_filter:
            print('Applying median filter...')
            for i in range(images.shape[-1]):
                sys.stdout.write('\rProgress: [{0:20s}] {1:.0f}%'.format('#' *
                                 int(20 * (i + 1) / images.shape[-1]),
                                 100 * ((i + 1) / images.shape[-1])))
                sys.stdout.flush()
                images[:, :, i] = medfilt(images[:, :, i], kernel_size=kernel)

        print('\nReconstructing...')
        for j in range(recon_height):
            sys.stdout.write('\rProgress: [{0:20s}] {1:.0f}%'.format('#' *
                             int(20 * (j + 1) / recon_height),
                             100 * ((j + 1) / recon_height)))
            sys.stdout.flush()
            sino_tmp = np.squeeze(images[j, :, :])
            
            if recon_alg is 'sart':
                image_tmp = iradon_sart(sino_tmp, theta=self.angles)
                for i in range(sart_iters - 1):
                    image_tmp = iradon_sart(sino_tmp, theta=self.angles, 
                                            image=image_tmp)
            else:
                image_tmp = iradon(sino_tmp, theta=self.angles, 
                                   filter=None, circle=True)
#            if crop_circle:
#                image_tmp = image_tmp[w0:wf, w0:wf]
                
            self.recon_data[:, :, j] = image_tmp
            if save:            
                save_folder = os.path.join(self.folder, 'reconstruction')
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                imsave(os.path.join(save_folder, '%04d.tif' % j), image_tmp)

