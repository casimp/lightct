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
from skimage import filters, measure
from skimage.transform import iradon, downscale_local_mean
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import time
#import cv2




class TomoScan(AnalyzeScan):
    
    def __init__(self, camera_port = 0):
        
        self.camera_port = camera_port
        
    def scan(self, proj = 100, wait = 0.1, save = True):
        
        camera = cv2.VideoCapture(self.camera_port)
        retval, im = camera.read()
        dims = im[:,:,2].shape + (proj, )
        self.im_stack = np.zeros(dims)
        
        for i in range(proj):
            retval, im = camera.read()
            self.im_stack[:, :, i] = skimage.color.rgb2hsv(im)[:, :, 2]
            time.sleep(wait)
        del(camera)

            
    def plot_histogram(self, proj = 0):
        
        histogram = np.histogram(self.im_stack[:, :, proj], 255)
        plt.plot(histogram[0])
        
        
    def auto_set_angles(self, thresh = 170, order = 25, plot = True):
        
        if thresh == 'otsu':
            thresh = filters.threshold_otsu(self.im_stack[:, :, 0])
        
        thresh_stack = (self.im_stack > thresh).astype(int)
        diff = [np.sum(np.abs(thresh_stack[:, :, i] - thresh_stack[:, :, 0])) \
                for i in range(thresh_stack.shape[-1])]
        minimas = argrelmin(np.array(diff), order = order)
        
        if plot:
            plt.figure()
            plt.plot(diff)
            plt.plot(minimas[0], np.array(diff)[minimas], 'r*')
            plt.plot([minimas[0][0], minimas[0][0]], [0, np.max(diff)], 'r--')
            plt.xlabel('Image number')
            plt.ylabel('Thresholded Pixels Relative to Image 1')
            plt.text(minimas[0][0], np.max(diff), r'$360^{\circ}$', 
                     horizontalalignment='center', verticalalignment='bottom')
        
        self.num_images = minimas[0][0]
        self.angle = np.linspace(0, 360, self.num_images)
            
    
    def manual_set_angles(self, interactive = True, num_images = None, 
                          ang_range = None):
        
        if interactive:
            backend = matplotlib.get_backend()
            err = ("Matplotlib running inline. Plot interaction not possible."
                   "\nTry running %matplotlib in the ipython console (and "
                   "%matplotlib inline to return to default behaviour). In "
                   "standard console use matplotlib.use('TkAgg') to interact.")
                     
            assert backend != 'module://ipykernel.pylab.backend_inline', err
            fig, ax_array = plt.subplots(1,2, figsize=(10, 5))
            
            ax_slider = plt.axes([0.2, 0.07, 0.5, 0.05])  
            ax_button = plt.axes([0.81, 0.05, 0.1, 0.075])
            
            ax_array[0].imshow(self.im_stack[:, :, 0])
            ax_array[1].imshow(self.im_stack[:, :, 0])
            ax_array[0].axis('off')
            ax_array[1].axis('off')
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.2)
            window_slider = Slider(ax_slider, 'Image', 0, 100, valinit = 0)
            store_button = Button(ax_button, r'Save - 360')
            
            def slider_update(val):
                ax_array[1].imshow(self.im_stack[:, :, int(window_slider.val)])
                window_slider.valtext.set_text('%i' % window_slider.val)
                fig.canvas.draw_idle()
                
            window_slider.on_changed(slider_update)
            
            def store_data(label):
                
                self.num_images = int(window_slider.val)
                self.angles = np.linspace(0, 360, int(window_slider.val))
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
        
        images = self.im_stack[:, :, :self.num_images]
        images = downscale_local_mean(images, downsample)
        
        if pre_filter != False:
            for i in range(images.shape[-1]): 
                images[:, :, i] = medfilt(images[:, :, i],kernel_size = kernel)
        
        recon_height, recon_width = images.shape[:2]
        self.recon_data=np.zeros((recon_width, recon_width, recon_height))

        for j in range(recon_height):
            sinotmp = np.squeeze(images[j, :, :])
            imagetmp = iradon(sinotmp, theta = self.angle, 
                              filter = None, circle = True)

            self.recon_data[:, :, j] = imagetmp
            if save:
                save_folder = os.path.join(self.folder, 'reconstruction')
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                sc.misc.imsave(save_folder + '/%04d.tif' % j, imagetmp)
    
    
    def vizualize(self, crop = 60, downsample = (2, 2, 2),  
                  kernel = 9, thresh = 'otsu'):
        
        data = self.recon_data[crop: -crop, crop: -crop, :]
        data = downscale_local_mean(data, downsample)
        
        for i in range(data.shape[2]):
            data[:, :, i] = medfilt(data[:, :, i], kernel_size = kernel)
        datathres = np.zeros(data.shape)

        if thresh == 'otsu':
            thresh = filters.threshold_otsu(data[:, :, 0]) + 0.08      
        
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

        
class LoadScan(object):
    
    def __init__(self, folder):
        self.folder = folder
        files =  [file for file in os.listdir(folder) if file[-4:] == '.tif']
        im_shape = sc.misc.imread(folder + files[0]).shape
        self.im_stack = np.zeros(im_shape + (len(files), ))
        for idx, file in enumerate(files):
            sys.stdout.write("\rProgress: [{0:20s}] {1:.0f}%".format('#' * 
            int(20*(idx + 1) / len(files)), 100*((idx + 1)/len(files))))
            sys.stdout.flush()
            self.im_stack[:, :, idx] = sc.misc.imread(folder + file)