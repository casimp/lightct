[![Build Status](https://travis-ci.org/casimp/lightct.svg?branch=master)](https://travis-ci.org/casimp/lightct)

lightct
=======

What is lightct?
----------------

Requirements
------------

lightct is built on Python’s scientific stack (numpy, scipy, matplotlib, scikit-image). Testing and development were carried out using the Anaconda (v 2.5.0) package manager, which built with the following versions:

-	Python: version 2.7.11, 3.5.1
-	numpy: version 1.10.4
-	scipy: version 0.17
-	matplotlib: version 1.5
-	scikit-image: version 0.11.3

The package also leverages opencv for the acquisition of images:

-	opencv: version 3.1

Installation
------------

Installing lightct is easily done using pip. Assuming you have pip installed, just run the following from the command-line:

```
pip install lightct
```

This command will download the latest version of lightct from the Python Package Index and install it to your system.

Alternatively, you can install from the distribution using the setup.py script. The source is stored in the GitHub repo, which can be browsed at:

https://github.com/casimp/lightct

Simply download and unpack, then navigate to the download directory and run the following from the command-line:

```
python setup.py install
```

Documentation
-------------

Documentation is hosted by readthedocs.
