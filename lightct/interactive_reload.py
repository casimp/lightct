# -*- coding: utf-8 -*-
"""
Created on Thu May 26 18:22:16 2016

@author: casimp
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import lightct

folder = input('Path to the projections ['']: ')
scan = lightct.LoadProjections(folder)

# Find number of projections in 360 degrees
correct = False
while not correct:
    est_nproj = input('Estimated number of projections in 360 [(50)]: ')
    est_nproj = 50 if est_nproj is '' else int(est_nproj)
    scan.auto_set_angles(est_nproj)
    correct = input('Is the analysis correct? [(y)/n])? ')
    correct = False if correct.lower() == 'n' else True
    
# Find number of projections in 360 degrees
correct = False
while not correct:
    window = input('Window to search for centre in [(400)]: ')
    window = 400 if window is '' else int(window)
    scan.auto_centre(window)
    correct = input('Is the analysis correct? [(y)/n])? ')
    correct = False if correct.lower() == 'n' else True

# Crop data
correct = False
while not correct:
    crop = input('Crop data (width, top, bottom) [0, 0, 0]: ')
    crop = (0, 0, 0) if crop is '' else tuple([int(i) for i in crop.split(',')])
    if crop == (0, 0, 0):
        correct = True
        print('Data not cropped')
    else:
        scan.set_crop(*crop)
        correct = input('Happy with the cropping? [(y)/n])? ')
        correct = False if correct.lower() == 'n' else True

# Reconstruct
downsample = input('Downsample ratio [(4, 4)]: ')
downsample = (4,4) if downsample is '' else tuple([int(i) for i in downsample.split(',')])
median_filter = input('Apply median filter to the projections? [y/(n)]: ')
median_filter = False if median_filter.lower() == 'n' or median_filter == '' else False
if median_filter:
    kernel = input('Kernel size [(5)]: ')
    kernel = 5 if window is '' else int(kernel)
    scan.reconstruct(downsample, crop=True, median_filter=median_filter, kernel=kernel)
else:
    scan.reconstruct(downsample, crop=True, median_filter=median_filter)

