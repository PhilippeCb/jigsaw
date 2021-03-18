#!/usr/bin/env python
"""
This file allows this repository to be installed using `pip -e`.
"""

from distutils.core import setup
from setuptools import find_packages

setup(name='jigsaw',
      version='latest',

      # package_dir only works in editable mode, when there is a folder with the package name in the root folder!
      # https://stackoverflow.com/questions/19602582/pip-install-editable-links-to-wrong-path
      # https://github.com/pypa/pip/issues/126
      # https://github.com/pypa/setuptools/issues/230

      # Automatically search and add all Python modules. This command adds all folders containing a __init__.py
      # to the Python modules.
      packages=find_packages(exclude=['test', 'test.*']),

      # The runtime dependencies of requirements.txt
      # (except dependencies for testing and development), should be listed in `install_requires` aswell. Whereas
      # requirements.txt contains strict dependency versions (using ==) this file should use relaxed versioning
      # using >= and <, to allow for future updates.
      # TODO: torch
      install_requires=[
          'numpy==1.13.1',
          'pillow==8.1.1',
      ],

      # Development requirements
      extras_require={
          'dev': [
              'pytest',
              'flake8'
          ]
      },
      )
