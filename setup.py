# from distutils.core import setup
# setup(name='pyShapeDetector', version='0.0.1dev', packages=['pyShapeDetector'])
from setuptools import setup, find_packages

setup(
    name="pyShapeDetector",
    version="0.0.1dev",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0,<2.0.0",  # Scipy cannot use Numpy 2.0.0 yet
        "open3d>=0.17.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.2.0",
    ],
    extras_require={
        "dev": [  # Optional dependencies for development
            "pytest>=6.0",  # Specific optional package with version constraint
        ],
        "triangulation": [
            "mapbox_earcut==1.0.1",
        ],
        "colors_cmap": [
            "matplotlib>=3.7.0",
        ],
        "h5_files": [
            "h5py==3.10.0",
        ],
        "e57_files": [
            "pye57",
        ],
        "lines_to_plane_with_shapely": [
            "shapely>=2.0.2",
        ],
    },
    author="Evandro Bernardes",
    author_email="evbernardes@gmail.com",
    description="Python 3 module to detect geometrical primitives in pointclouds",
    license="GPL-3.0-or-later",
    url="https://github.com/evbernardes/pyShapeDetector",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
