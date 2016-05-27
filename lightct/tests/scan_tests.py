# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:25:14 2016

@author: casim
"""

from lightct.tomo_scan import TomoScan
from mock import patch
import numpy as np

@patch("lightct.tomo_scan.image_acquisition")
def testfunction(mfun):
    mfun.return_value  = np.random.rand(20, 10, 3)
    TomoScan(10, '', save=False)
