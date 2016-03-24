# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from mayavi import mlab
import sys
import os
import numpy as np
import scipy as sc
from scipy.signal import medfilt, argrelmin
from skimage import filters, measure, color
from skimage.transform import iradon, downscale_local_mean
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import time
import cv2


class TomoScan(object):
    
    def __init__(self, camera_port = 0):
        
        self.camera_port = camera_port
        
    def scan(self, proj = 100, wait = 0.1, save = True):
        
        camera = cv2.VideoCapture(self.camera_port)
        retval, im = camera.read()
        dims = im[:,:,2].shape + (proj, )
        self.im_stack = np.zeros(dims)
        
        for i in range(proj):
            retval, im = camera.read()
            self.im_stack[:, :, i] = color.rgb2hsv(im)[:, :, 2]
            time.sleep(wait)
        del(camera)
        self.proj_width = self.im_stack.shape[0]
        self.proj_height = self.im_stack.shape[1]
        

            
    def plot_histogram(self, proj = 0):
        
        histogram = np.histogram(self.im_stack[:, :, proj], 255)
        plt.plot(histogram[0])
        
        
       
    def auto_set_angles(self, proj_ref = 5, order = 25, plot = True):
        """
        Attempts to automatically locate image at 360 degrees (and multiples
        of 360 degrees). Alignment based on difference calculation between 
        reference projection each subsequent projections. 
        
        # proj_ref:   Projection to use as initial or reference projection.
                      Recommended to be greater than 1 (due to acquisiton 
                      spacing issues in intital projections)   
        # order:      Window in which to search for minimas in the
                      difference calculations - should be approx equal to 
                      (number of projections in 360) / 2
        # plot:       Plot the difference results
        """
        ref = downscale_local_mean(self.im_stack[:, :, proj_ref], (3, 3))
        diff = np.nan * np.ones((self.im_stack.shape[-1] - proj_ref))
        proj_nums = range(proj_ref, self.im_stack.shape[-1])
        for idx, i in enumerate(proj_nums):
            current = downscale_local_mean(self.im_stack[:, :, i], (3, 3))
            tmp = current - ref
            diff[idx] = tmp.std()
        minimas = argrelmin(np.array(diff), order = order)
        print(minimas)
        self.p0 = proj_ref
        self.num_images = minimas[0][0] + 1
        self.angles = np.linspace(0, 360, self.num_images, dtype = int)
        
        if plot:
            plt.plot(proj_nums, diff)
            plt.plot(minimas[0] + proj_ref, np.array(diff)[minimas], 'r*')
            plt.plot([minimas[0][0] + proj_ref, minimas[0][0] + proj_ref], [0, np.max(diff)], 'r--')
            plt.xlabel('Image number')
            plt.ylabel('Thresholded Pixels Relative to Image 1')
            plt.text(minimas[0][0] + proj_ref, np.max(diff), r'$360^{\circ}$', 
                     horizontalalignment='center', verticalalignment='bottom')
                     
        print('\n%i images in a 360 rotation. \n\n If this is incorrect '
              'either rerun with a different value for order or use the manual'
              ' method.' % self.num_images)
        
    def auto_centre(self, window = 200, cropped = False):
        if cropped:
            ref = self.cropped[:, window:-window, self.p0]
            im_180 = self.cropped[:, :, int(self.num_images / 2) + self.p0]
        else:
            ref = self.im_stack[:, window:-window, self.p0]
            im_180 = self.im_stack[:, :, int(self.num_images / 2) + self.p0]
        flipped = np.fliplr(im_180)
        win_range = range(-window, window)
        diff = np.nan * np.zeros(len(win_range))
        
        for idx, i in enumerate(win_range):
            
            cropped = flipped[:, window + i: -window + i]
            tmp = cropped - ref
            diff[idx] = tmp.std()
        
        
        minima = np.argmin(diff)
        self.cor = win_range[minima]
        plt.plot(win_range, diff)
        plt.figure()
        plt.imshow(flipped[:,  window + win_range[minima]: -window + win_range[minima]] - ref)
        
        
    def crop_to_centre(self):
        if self.cor < 0:
            self.cropped = self.im_stack[:, self.cor:, :]
        else:
            self.cropped = self.im_stack[:, :-self.cor, :]       
        
    
    def manual_set_angles(self, interact = True, proj_ref = 5, 
                          num_images = None, ang_range = None):
        """
        Manually define the number of images in 360 degrees. Defaults to 
        interactive mode in which images can be compared against initial, 
        reference image.
        
        # interact:   Run in interactive mode (True/False)
        # proj_ref:   Projection to use as initial or reference projection.
                      Recommended to be greater than 1 (due to acquisiton 
                      spacing issues in intital projections)   
        # num_images: If not in interact mode, manually specify number 
                      of images
        # ang_range:  If not in interact mode, manually specify angular range 
                      of images (must be multiple of 180)
        """
        if interact:
            backend = matplotlib.get_backend()
            err = ("Matplotlib running inline. Plot interaction not possible."
                   "\nTry running %matplotlib in the ipython console (and "
                   "%matplotlib inline to return to default behaviour). In "
                   "standard console use matplotlib.use('TkAgg') to interact.")
                     
            assert backend != 'module://ipykernel.pylab.backend_inline', err
            fig, ax_array = plt.subplots(1,2, figsize=(10, 5))
            
            ax_slider = plt.axes([0.2, 0.07, 0.5, 0.05])  
            ax_button = plt.axes([0.81, 0.05, 0.1, 0.075])
            
            ax_array[0].imshow(self.im_stack[:, :, proj_ref])
            ax_array[1].imshow(self.im_stack[:, :, proj_ref])
            ax_array[0].axis('off')
            ax_array[1].axis('off')
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.2)
            nfiles = self.im_stack.shape[-1] + 1
            window_slider = Slider(ax_slider, 'Image', proj_ref, nfiles, valinit = 0)
            store_button = Button(ax_button, r'Save - 360')
            
            def slider_update(val):
                ax_array[1].imshow(self.im_stack[:, :, int(window_slider.val)])
                window_slider.valtext.set_text('%i' % window_slider.val)
                fig.canvas.draw_idle()
                
            window_slider.on_changed(slider_update)
            
            def store_data(label):
                ### Check this is correct - proj_ref!!!
                self.num_images = int(window_slider.val) - proj_ref + 1
                self.angles = np.linspace(0, 360, self.num_images)
                self.p0 = proj_ref
                plt.close()
                
            store_button.on_clicked(store_data)
            return window_slider, store_button
        
        else:
            error = ('Images must cover a rotational range of 180 or 360 deg')
            assert (ang_range == 180) or (ang_range == 360), error
            self.angles = np.linspace(0, ang_range, num_images)
            self.imstack = self.imstack[:, :, :num_images]
        
        
    def reconstruct(self, downsample = (4, 4, 1), pre_filter = True, 
                    kernel = 9, save = True):
        
        if self.cor <= 0:
            images = self.im_stack[:, self.cor:, self.p0:self.num_images + self.p0]
        else:
            images= self.im_stack[:, :-self.cor, self.p0:self.num_images + self.p0]    
            
        images = downscale_local_mean(images, downsample)
        
        if pre_filter != False:
            for i in range(images.shape[-1]): 
                images[:, :, i] = medfilt(images[:, :, i],kernel_size = kernel)
        
        recon_height, recon_width = images.shape[:2]
        self.recon_data=np.zeros((recon_width, recon_width, recon_height))

        for j in range(recon_height):
            sinotmp = np.squeeze(images[j, :, :])
            imagetmp = iradon(sinotmp, theta = self.angles, 
                              filter = None, circle = True)

            self.recon_data[:, :, j] = imagetmp
            if save:
                save_folder = os.path.join(self.folder, 'reconstruction')
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                sc.misc.imsave(save_folder + '/%04d.tif' % j, imagetmp)

    def recon_threshold(self, image = 0):
        backend = matplotlib.get_backend()
        err = ("Matplotlib running inline. Plot interaction not possible."
               "\nTry running %matplotlib in the ipython console (and "
               "%matplotlib inline to return to default behaviour). In "
               "standard console use matplotlib.use('TkAgg') to interact.")
                 
        assert backend != 'module://ipykernel.pylab.backend_inline', err
        
        fig, ax_array = plt.subplots(1,2, figsize=(12, 5))
        
        bins = np.max(self.recon_data[:, :, image]).astype(int)
        #histogram = np.histogram(self.recon_data[:, :, image], bins)[0]
        
        ax_slider = plt.axes([0.2, 0.07, 0.5, 0.05])  
        ax_button = plt.axes([0.81, 0.05, 0.1, 0.075])
        
        #ax_array[0].plot(histogram)
        #line, = ax_array[0].plot([0, 0], [0, np.max(histogram)], 'r-.')
        ax_array[0].imshow(self.recon_data[:, :, image])
        ax_array[1].imshow(self.recon_data[:, :, image] > 0)
        ax_array[1].axis('off')
        ax_array[0].axis('off')
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        window_slider = Slider(ax_slider, 'Thresh', 0, bins, valinit = 0)
        store_button = Button(ax_button, r'Save')
        
        def slider_update(val):
            ax_array[1].imshow(self.recon_data[:,:, image] > window_slider.val)
            window_slider.valtext.set_text('%i' % window_slider.val)
            #line.set_xdata([window_slider.val,  window_slider.val])
            fig.canvas.draw_idle()
            
        window_slider.on_changed(slider_update)
        
        def store_data(label):
            
            self.thresh = int(window_slider.val)
            plt.close()
            
        store_button.on_clicked(store_data)
        return window_slider, store_button    
        
    
    def vizualize(self, crop = 60, downsample = (2, 2, 2),  
                  kernel = 9, thresh = None):
        
        data = self.recon_data[crop: -crop, crop: -crop, :]
        data = downscale_local_mean(data, downsample)
        
        for i in range(data.shape[2]):
            data[:, :, i] = medfilt(data[:, :, i], kernel_size = kernel)
        datathres = np.zeros(data.shape)
        
        if thresh == None:
            try: 
                thresh = self.thresh
            except AttributeError:
                error = ('Either manually define thresh variable or run '
                         'recon_threshold method.')
                raise AttributeError(error)
        elif thresh == 'otsu':
            thresh = filters.threshold_otsu(data)    
        
        for i in range(data.shape[2]):
            if (np.max(data[:, :, i])) >= 0.35:
                datathres[:, :, i]=(data[:, :, i] >= thresh)
            else :
                datathres[:,:,i] = np.zeros(data.shape[:2])

        verts, faces = measure.marching_cubes(datathres, 0)
        mlab.triangular_mesh([vert[0] for vert in verts],
                             [vert[1] for vert in verts],
                             [vert[2] for vert in verts], faces)
        mlab.show()

        
class LoadScan(TomoScan):
    
    def __init__(self, folder):
        self.folder = folder
        files =  [f for f in os.listdir(folder) if f[-4:] == '.tif']
        im_shape = sc.misc.imread(os.path.join(self.folder, files[0])).shape
        self.im_stack = np.zeros(im_shape + (len(files), ))
        for idx, fname in enumerate(files):
            sys.stdout.write("\rProgress: [{0:20s}] {1:.0f}%".format('#' * 
            int(20*(idx + 1) / len(files)), 100*((idx + 1)/len(files))))
            sys.stdout.flush()
            f = os.path.join(self.folder, fname)
            self.im_stack[:, :, idx] = sc.misc.imread(f)
        self.cor = 0