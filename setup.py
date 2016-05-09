try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='lightct',
    version='0.0.1',
    author='L.Courtois, C. Simpson',
    author_email='c.a.simpson01@gmail.com',
    packages=['lightct'],
    install_requires=[cv2, matplotlib, numpy, scipy, skimage],
    url='https://github.com/casimp/lightct',
    license='LICENSE.txt',
    description='Visual light computed tomography.',
    keywords = ['ct', 'vlct', 'computed tomography'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows"]
)
