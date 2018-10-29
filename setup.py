#!/usr/bin/env python
"""
This file allows this repository to be installed using `pip -e`.
"""

from distutils.core import setup
from setuptools import find_packages

setup(name='jigsaw',
      version='latest',
      description='Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles',
      author='Philippe Chiberre',
      packages=find_packages(exclude=['test', 'test.*']),
     )
