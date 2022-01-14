# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2012, 2014, 2015  Esben S. Nielsen
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

# workaround python bug: http://bugs.python.org/issue15881#msg170215
# remove when python 2 support is dropped
"""The setup module."""
import multiprocessing  # noqa: F401
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

import versioneer

requirements = ['setuptools>=3.2', 'pyproj>=2.2', 'configobj',
                'pykdtree>=1.3.1', 'pyyaml', 'numpy>=1.10.0',
                ]

if sys.version_info < (3, 10):
    requirements.append('importlib_metadata')

test_requires = ['rasterio', 'dask', 'xarray', 'cartopy', 'pillow', 'matplotlib', 'scipy', 'zarr',
                 'pytest-lazy-fixtures']
extras_require = {'numexpr': ['numexpr'],
                  'quicklook': ['matplotlib', 'cartopy>=0.20.0', 'pillow'],
                  'rasterio': ['rasterio'],
                  'dask': ['dask>=0.16.1'],
                  'cf': ['xarray'],
                  'gradient_search': ['shapely'],
                  'xarray_bilinear': ['xarray', 'dask', 'zarr'],
                  'tests': test_requires}

setup_requires = ['numpy>=1.10.0', 'cython']
test_requires = ['rasterio', 'dask', 'xarray', 'cartopy>=0.20.0', 'pillow', 'matplotlib', 'scipy', 'zarr',
                 'pytest-lazy-fixture']

if sys.platform.startswith("win"):
    extra_compile_args = []
else:
    extra_compile_args = ["-O3", "-Wno-unused-function"]

extensions = [
    Extension("pyresample.ewa._ll2cr", sources=["pyresample/ewa/_ll2cr.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=extra_compile_args),
    Extension("pyresample.ewa._fornav",
              sources=["pyresample/ewa/_fornav.pyx",
                       "pyresample/ewa/_fornav_templates.cpp"],
              include_dirs=[np.get_include()],
              language="c++", extra_compile_args=extra_compile_args,
              depends=["pyresample/ewa/_fornav_templates.h"]),
    Extension("pyresample.gradient._gradient_search",
              sources=["pyresample/gradient/_gradient_search.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=extra_compile_args),
]

cmdclass = versioneer.get_cmdclass()

entry_points = {
    "pyresample.resamplers": [
        "nearest = pyresample.future.resamplers.nearest:KDTreeNearestXarrayResampler",
    ],
}

if __name__ == "__main__":
    README = open('README.md', 'r').read()
    setup(name='pyresample',
          url='https://github.com/pytroll/pyresample',
          version=versioneer.get_version(),
          cmdclass=cmdclass,
          description='Geospatial image resampling in Python',
          long_description=README,
          long_description_content_type='text/markdown',
          author='Thomas Lavergne',
          author_email='t.lavergne@met.no',
          package_dir={'pyresample': 'pyresample'},
          packages=find_packages(),
          package_data={'pyresample.test': ['test_files/*']},
          python_requires='>=3.7',
          setup_requires=setup_requires,
          install_requires=requirements,
          extras_require=extras_require,
          ext_modules=cythonize(extensions),
          entry_points=entry_points,
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
