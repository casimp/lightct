# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:25:14 2016

@author: casim
"""

from nose.tools import *
from lightct.load_scan import LoadProjections
import os


proj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'test_scan\projections')

def test_load():
    scan_data = LoadProjections(proj_path)
#    assert_equal(gold.name, "GoldRoom")
#    assert_equal(gold.paths, {})

def test_auto_angles():
    scan_data = LoadProjections(proj_path)
    scan_data.auto_set_angles(20, plot=False)
    
def test_set_angles():
    scan_data = LoadProjections(proj_path)
    scan_data.set_angles(21, 360)

    
def test_auto_centre():
    scan_data = LoadProjections(proj_path)
    scan_data.auto_centre(plot=False)
#
#def test_map():
#    start = Room("Start", "You can go west and down a hole.")
#    west = Room("Trees", "There are trees here, you can go east.")
#    down = Room("Dungeon", "It's dark down here, you can go up.")
#
#    start.add_paths({'west': west, 'down': down})
#    west.add_paths({'east': start})
#    down.add_paths({'up': start})
#
#    assert_equal(start.go('west'), west)
#    assert_equal(start.go('west').go('east'), start)
#    assert_equal(start.go('down').go('up'), start)

