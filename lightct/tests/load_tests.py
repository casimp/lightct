# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:25:14 2016

@author: casim
"""

from nose.tools import assert_equal
from lightct.load_scan import LoadProjections
import os
from mock import patch


proj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'test_scan/projections')

def test_load():
    """
    Check that the correct number of files is being loaded.
    """
    scan_data = LoadProjections(proj_path)
    num_tifs = len([i for i in os.listdir(proj_path) if i[-4:] == '.tif'])
    assert_equal(scan_data.im_stack.shape[-1], num_tifs)


@patch("matplotlib.pyplot.show")
def test_auto_angles(mock_show):
    scan_data = LoadProjections(proj_path)
    scan_data.auto_set_angles(20)
    
    
@patch("matplotlib.pyplot.show")
def test_manual_angles(mock_show):
    scan_data = LoadProjections(proj_path)
    scan_data.manual_set_angles()
    

def test_set_angles():
    scan_data = LoadProjections(proj_path)
    scan_data.set_angles(21, 360)
    
    
@patch("matplotlib.pyplot.show")
def test_auto_centre(mock_show):
    scan_data = LoadProjections(proj_path)
    scan_data.set_angles(21, 360)
    scan_data.auto_centre()
    
def test_set_centre():
    scan_data = LoadProjections(proj_path)
    scan_data.set_angles(21, 360)
    scan_data.set_centre(-40)
    assert_equal(scan_data.cor_offset, -40)


@patch("matplotlib.pyplot.show")
def test_set_crop(mock_show):
    scan_data = LoadProjections(proj_path)
    scan_data.set_angles(21, 360)
    scan_data.set_centre(-40)
    
    assert_equal(scan_data.crop, (None, None, None, None))
    scan_data.set_crop(100, 100, 100)
    assert_equal(scan_data.crop, (100, -100, 100, -100))
    scan_data.set_crop(0, 0, 0, plot=False)
    assert_equal(scan_data.crop, (None, None, None, None))
    
    
def test_reconstruction():
    scan_data = LoadProjections(proj_path)
    scan_data.set_angles(21, 360)
    scan_data.set_centre(-40)
    scan_data.set_crop(300, 100, 100, plot=False)
    scan_data.reconstruct(downsample=(4,4), crop=True, 
                          median_filter=True, kernel=3, save=False)
    scan_data.reconstruct(downsample=(8,8), crop=False, 
                          median_filter=False, save=True)
