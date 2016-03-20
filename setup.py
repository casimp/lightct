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
    url='https://github.com/casimp/lightct',
    license='LICENSE.txt',
    description='Visual light computed tomography.',
    keywords = ['ct', 'vlct', 'computed tomography'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows"]
)