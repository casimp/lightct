lightct
=======

[![Build Status](https://travis-ci.org/casimp/lightct.svg?branch=master)](https://travis-ci.org/casimp/lightct) [![Coverage Status](https://coveralls.io/repos/github/casimp/lightct/badge.svg?branch=master)](https://coveralls.io/github/casimp/lightct?branch=master)

What is lightct?
----------------

lightct is a python package for visual light computed tomography. With just a webcam and a rotating stage the package will allow for the acquisition of 2D projections and the reconstruction of objects in 3D.

This package will be showcased at the Royal Society Summer Exhibition (4th - 10th July 2016), allowing members of the public to run their own synchrotron style experiment:

https://royalsociety.org/events/summer-science-exhibition/exhibits/4d-science/

Requirements
------------

lightct is built on Python’s scientific stack (numpy, scipy, matplotlib, scikit-image). Development was carried out using the Anaconda (v 2.5.0) package manager, which is the recommended route for the installation of Python and the scientific stack:

https://www.continuum.io/downloads

lightct works with both Python 2 and 3 and has been tested under the following conditions:

-	Python: version 2.7, 3.4+
-	numpy: version 1.10
-	scipy: version 0.17
-	matplotlib: version 1.5
-	scikit-image: version 0.11

The package also leverages opencv for the acquisition of images:

-	opencv: version 3.1

OpenCV can be installed on an Anaconda flavoured build of python by running the following in the command prompt:

```
conda install -c https://conda.binstar.org/menpo opencv3
```

Installation of lightct
-----------------------

Installing lightct is easily done using pip. Assuming you have pip installed (included with the Anaconda build), just run the following from the command-line:

```
pip install lightct
```

This command will download the stable version of lightct from the Python Package Index and install it to your system.

Alternatively, you can install from the most recent distribution using the setup.py script. The source is stored in the GitHub repo, which can be browsed at:

https://github.com/casimp/lightct

Simply download and unpack, then navigate to the download directory and run the following from the command-line:

```
python setup.py install
```

Documentation
-------------

Please see the lightct-guide for more information:

https://github.com/casimp/lightct/blob/master/docs/lightct_guide.ipynb
