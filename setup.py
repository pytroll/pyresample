#pyresample, Resampling of remote sensing image data in python
# 
#Copyright (C) 2012  Esben S. Nielsen
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup
import sys
import os

import imp

version = imp.load_source('pyresample.version', 'pyresample/version.py')

requirements = ['pyproj', 'numpy', 'scipy', 'configobj']
if sys.version_info < (2, 6):
    # multiprocessing is not in the standard library
    requirements.append('multiprocessing')

setup(name='pyresample',
      version=version.__version__,
      description='Resampling of remote sensing data in Python',
      author='Esben S. Nielsen',
      author_email='esn@dmi.dk',
      package_dir = {'pyresample': 'pyresample'},
      packages = ['pyresample'],      
      install_requires=requirements,
      zip_safe = False,
      classifiers=[
      'Development Status :: 5 - Production/Stable',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Programming Language :: Python',
      'Operating System :: OS Independent',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering'
      ]
      )



