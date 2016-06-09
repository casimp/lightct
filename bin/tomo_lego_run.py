# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:30:18 2016

@author: lbq76018
"""
import lightct
import sys
import numpy as np
import os

working_dir = os.path.split(sys.argv[0])[0]
sample_name = sys.argv[1]

proj_folder = os.path.join(working_dir, os.path.join(sample_name, 'projections'))
info_file = os.path.join(working_dir, os.path.join(sample_name, r'info.txt'))

data_id = ['n_acq', 'n_est', 'n_proj', 'centre_win', 'cor', 
           'crop_width', 'crop_top', 'crop_bottom', 'acquire', 'calc_angles', 
           'calc_centre', 'set_crop', 'downsample', 'reconstruct', 'plot_ang', 
           'plot_centre', 'plot_crop', 'im_dims1', 'im_dims2', 'im_dims3']

with open(info_file, 'r') as f:
    lines = []
    for line in f:
        lines.append(line.rstrip().split(' # '))
        
data_vals = np.array([line[0] for line in lines[1:]], dtype='float64')
data = dict(zip(data_id, data_vals))
  
# Acquire or load projections
if data['acquire']:
    scan = lightct.TomoScan(int(data['n_acq']), proj_folder)
    lines[data_id.index('acquire') + 1][0] = str(int(0))
else:
    print('Loading projections')
    scan = lightct.LoadProjections(proj_folder)
    
# Number of angles in 360
if data['calc_angles']:
    print('Setting angles')
    scan.auto_set_angles(int(data['n_est']), plot=data['plot_ang'])
    lines[data_id.index('n_proj') + 1][0] = str(int(scan.num_images))
else:
    scan.set_angles(int(data['n_proj']))
    
# Centre of rotation
if data['calc_centre']:
    scan.auto_centre(int(data['centre_win']), plot=data['plot_centre'])
    lines[data_id.index('cor') + 1][0] = str(int(scan.cor_offset))
else:
    scan.set_centre(int(data['cor']))
    
# Projection cropping
if data['set_crop']:
    scan.set_crop(int(data['crop_width']), int(data['crop_top']), 
                  int(data['crop_bottom']), plot=data['plot_crop'])
else:
    scan.set_crop(int(data['crop_width']), int(data['crop_top']),
                  int(data['crop_bottom']), plot=False)
                  
if data['reconstruct']:
    down_ratio = (int(data['downsample']), ) *2
    scan.reconstruct(down_ratio)
    lines[data_id.index('im_dims1') + 1][0] = str(int(scan.recon_data.shape[0]))
    lines[data_id.index('im_dims2') + 1][0] = str(int(scan.recon_data.shape[1]))
    lines[data_id.index('im_dims3') + 1][0] = str(int(scan.recon_data.shape[2]))

with open(info_file, 'w') as f:
            
    for idx, line in enumerate(lines):
        line_comb = ' # '.join([str(i) for i in line])
        f.write('%s\n' % line_comb)
