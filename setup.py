# from distutils.core import setup
# setup(name='pyShapeDetector', version='0.0.1dev', packages=['pyShapeDetector'])
from setuptools import setup, find_packages

setup(
    name="pyShapeDetector",
    version="0.0.1dev",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.25.2",
        "open3d>=0.17.0",
    ],
    author="Evandro Bernardes",
    author_email="evbernardes@gmail.com",
    description="Python 3 module to detect geometrical primitives in pointclouds",
    license="MIT",
    url="https://github.com/evbernardes/pyShapeDetector",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
