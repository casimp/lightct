try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='lightct',
    version='0.0.2',
    author='C. Simpson, L.Courtois',
    author_email='c.a.simpson01@gmail.com',
    packages=['lightct'],
    url='https://github.com/casimp/lightct',
    scripts=['bin/interactive_reload.py', 'bin/interactive_lightct.py'],
    license='LICENSE.txt',
    description='Visual light computed tomography.',
    keywords=['ct', 'vlct', 'computed tomography'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows"]
)
