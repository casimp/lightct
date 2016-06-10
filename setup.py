try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('description') as readme_file:
    long_description = readme_file.read()

setup(
    name='lightct',
    version='0.1.0',
    author='C. Simpson, L.Courtois',
    author_email='c.a.simpson01@gmail.com',
    packages=['lightct'],
    url='https://github.com/casimp/lightct',
    download_url='https://github.com/casimp/lightct/tarball/0.1',
    scripts=['bin/interactive_reload.py', 'bin/interactive_lightct.py',
             'bin/tomo_lego_run.py'],
    license='LICENSE.txt',
    description='Visual light computed tomography.',
    long_description=long_description,
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
