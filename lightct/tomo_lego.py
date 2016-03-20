from __future__ import print_function, division

from mayavi import mlab

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


PREFIX = 'snowman_'
nproj = 20
nstep = nproj+1

print(nstep,nproj)

angle = np.linspace(0,360,nstep*2,dtype=int)


camera_port = 0
camera = cv2.VideoCapture(0)
start = time.clock()
for i in range (20) :
    calibi=angle[i]
    retval,im = camera.read()
    im = (skimage.color.rgb2hsv(im))[:,:,2]
end=time.clock()
del(camera)
width = im.shape[1]
height = im.shape[0]
overhead = (end-start)/50



time_half_rotation = 3.2
#time_half_rotation = int(raw_input('Time for 180 rotation (s) : '))


tproj = time_half_rotation/nproj

if tproj<(overhead*1):
    raise Exception('Too many projections')


projection=np.zeros((height,width,nstep*2))

camera = cv2.VideoCapture(camera_port)
for i in range (nstep*2) :
    temp = angle[i]
    retval,imtmp=camera.read()
    projection[:,:,i] = (skimage.color.rgb2hsv(imtmp))[:,:,2]
#    projection[:,:,i,0]=imtmp[:,:,0]
#    projection[:,:,i,1]=imtmp[:,:,1]
#    projection[:,:,i,2]=imtmp[:,:,2]
    time.sleep((tproj-overhead))
del(camera)

downsample = skimage.transform.downscale_local_mean(projection,(4,4,1))
for i in range (nstep*2) :
    downsample[:,:,i] = sc.signal.medfilt(downsample[:,:,i],kernel_size=9)

down_height = downsample.shape[0]
down_width = downsample.shape[1]

data=np.zeros((down_width,down_width,down_height))
for j in range (0,down_height) :
    sinotmp = np.squeeze(downsample[j,:,:])
    imagetmp = iradon(sinotmp,theta=angle,filter=None,circle=True)
   # imagetmp = iradon_sart(sinotmp, theta = angle)
    #imagetmp2=iradon_sart(sinotmp, theta = angle,image=imagetmp)
    #imagetmp3=iradon_sart(sinotmp, theta = angle,image=imagetmp2)
    #imagetmp4=iradon_sart(sinotmp, theta = angle,image=imagetmp3)
    sc.misc.imsave('/Users/lcourtois/Documents/Perso/3Dmagination/Software/TomoLego/FBP_'+'%04d' % j+'.tif',imagetmp)
    data[:,:,j]=imagetmp
    
    
    
    
    
data = data[60:down_width-60,60:down_width-60,:]
datafilter=skimage.transform.downscale_local_mean(data,(2,2,2))
for i in range (0,datafilter.shape[2]) :
    datafilter[:,:,i]=sc.signal.medfilt(datafilter[:,:,i],kernel_size=9)



datathres=np.zeros((datafilter.shape[0],datafilter.shape[1],datafilter.shape[2]))
for i in range (0,datafilter.shape[2]) :
    tempotsu=threshold_otsu(datafilter[:,:,i])+0.08
    if ((np.max(datafilter[:,:,i]))>=0.35) :
        datathres[:,:,i]=(datafilter[:,:,i]>=tempotsu)
    else :
        datathres[:,:,i]=np.zeros((datafilter.shape[0],datafilter.shape[1]))
#datatemp = sc.ndimage.morphology.binary_opening(datathres, iterations=7)
#datatemp = sc.ndimage.morphology.binary_closing(datatemp, iterations=5)
verts, faces = measure.marching_cubes(datathres, 0)
mlab.triangular_mesh([vert[0] for vert in verts],[vert[1] for vert in verts],[vert[2] for vert in verts],faces)
mlab.show()

#datatemp=datatemp *10
#src = mlab.pipeline.scalar_field(datatemp)
#mlab.pipeline.iso_surface(src, contours=5, opacity=1)

