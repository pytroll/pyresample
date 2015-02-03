# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2012, 2014, 2015  Esben S. Nielsen
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

# workaround python bug: http://bugs.python.org/issue15881#msg170215
import multiprocessing
from setuptools import setup
import sys

import imp

version = imp.load_source('pyresample.version', 'pyresample/version.py')

requirements = ['pyproj', 'numpy', 'configobj']
extras_require = {'pykdtree': ['pykdtree'],
                  'numexpr': ['numexpr'],
                  'quicklook': ['matplotlib', 'basemap']}

if sys.version_info < (2, 6):
    # multiprocessing is not in the standard library
    requirements.append('multiprocessing')

setup(name='pyresample',
      version=version.__version__,
      description='Resampling of remote sensing data in Python',
      author='Thomas Lavergne',
      author_email='t.lavergne@met.no',
      package_dir={'pyresample': 'pyresample'},
      packages=['pyresample'],
      install_requires=requirements,
      extras_require=extras_require,
      test_suite='pyresample.test.suite',
      zip_safe=False,
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
          'Programming Language :: Python',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering'
      ]
      )
