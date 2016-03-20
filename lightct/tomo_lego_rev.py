from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from mayavi import mlab

import os
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale
from skimage.transform import iradon
from PIL import Image
from skimage.transform import iradon_sart
import cv2
import time
import skimage
from skimage import color
from skimage import measure
from skimage.filters import threshold_otsu
from skimage import img_as_ubyte

class TomoScan(object):
    
    def __init__(self, camera_port = 0, nproj = 20, ang_range = [0, 360]):
        
        self.camera_port = camera_port
        self.nproj = nproj
        self.nstep = nproj + 1
        self.angle = np.linspace(ang_range[0], ang_range[1], self.nstep * 2, dtype = int)
        print(self.nstep, self.nproj)
        
    def image_timings(self):


        camera = cv2.VideoCapture(self.camera_port)
        start = time.clock()
        for i in range (20) :
            retval,im = camera.read()
            im = (skimage.color.rgb2hsv(im))[:,:,2]
        end=time.clock()
        del(camera)
        self.width = im.shape[1]
        self.height = im.shape[0]
        self.overhead = (end-start) / 50

        #time_half_rotation = int(raw_input('Time for 180 rotation (s) : '))
        time_half_rotation = 3.2 ######!!!!!
        self.tproj = time_half_rotation / self.nproj
        
        if self.tproj<(self.overhead * 1):
            raise Exception('Too many projections')
        
    def scan(self, reconstruct = True, save = True):
        
        self.projection=np.zeros((self.height, self.width, self.nstep*2))
        camera = cv2.VideoCapture(self.camera_port)
        for i in range (self.nstep * 2) :
            #temp = self.angle[i]
            retval, imtmp = camera.read()
            self.projection[:, :, i] = (skimage.color.rgb2hsv(imtmp))[:, :, 2]

            time.sleep((self.tproj - self.overhead))
        del(camera)
        
        if reconstruct:
            self.reconstruct()
    
    def load_projs(self, folder):
        
        files =  [file for file in os.listdir(folder) if file[-4:] == '.tif']
        self.im_stack = np.zeros((sc.misc.imread(folder + files[0]).shape + (len(files), )))
            
        for idx, file in enumerate(files):
            self.im_stack[:, :, idx] = sc.misc.imread(folder + file) 
        
    def reconstruct(self, downsample = (4, 4, 1), pre_filter = 'median', kernel = 9, save_folder = ''):
        
        downsample = skimage.transform.downscale_local_mean(self.projection, downsample)
        
        if pre_filter != False:
            pre_filter = sc.signal.medfilt if pre_filter == 'median' else pre_filter
            for i in range (self.nstep * 2): 
                downsample[:, :, i] = sc.signal.medfilt(downsample[:, :, i], kernel_size = kernel)
        
        down_height = downsample.shape[0]
        down_width = downsample.shape[1]
        
        self.data = np.zeros((down_width,down_width,down_height))
        
        for j in range (0, down_height) :
            sinotmp = np.squeeze(downsample[j, :, :])
            imagetmp = iradon(sinotmp, theta = self.angle, 
                              filter = None, circle = True)

            sc.misc.imsave('/Users/lcourtois/Documents/Perso/3Dmagination/Software/TomoLego/FBP_'+'%04d' % j+'.tif',imagetmp)
            self.data[:, :, j] = imagetmp
    
    def vizualize(self, crop = 60, downsample = (2, 2, 2), filter = True, kernel = 9):
        
        self.data = self.data[crop: -crop, crop: -crop, :]
        
        datafilter = skimage.transform.downscale_local_mean(self.data, downsample)
        if filter:
            for i in range(datafilter.shape[2]):
                datafilter[:, :, i]=sc.signal.medfilt(datafilter[:, :, i], kernel_size = kernel)
        
        
        
        datathres=np.zeros((datafilter.shape[0],datafilter.shape[1],datafilter.shape[2]))
        for i in range (0,datafilter.shape[2]) :
            tempotsu=threshold_otsu(datafilter[:, :, i]) + 0.08
            if (np.max(datafilter[:, :, i])) >= 0.35:
                datathres[:, :, i]=(datafilter[:, :, i] >= tempotsu)
            else :
                datathres[:,:,i] = np.zeros((datafilter.shape[0], datafilter.shape[1]))

        verts, faces = measure.marching_cubes(datathres, 0)
        mlab.triangular_mesh([vert[0] for vert in verts],[vert[1] for vert in verts],[vert[2] for vert in verts],faces)
        mlab.show()
        
        #datatemp=datatemp *10
        #src = mlab.pipeline.scalar_field(datatemp)
        #mlab.pipeline.iso_surface(src, contours=5, opacity=1)

