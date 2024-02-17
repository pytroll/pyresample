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
"""The setup module."""
import sys

import numpy as np
from Cython.Build import build_ext
from Cython.Distutils import Extension
from setuptools import find_packages, setup

import versioneer

requirements = ['setuptools>=3.2', 'pyproj>=3.0', 'configobj',
                'pykdtree>=1.3.1', 'pyyaml', 'numpy>=1.21.0',
                "shapely", "donfig", "platformdirs",
                ]

if sys.version_info < (3, 10):
    requirements.append('importlib_metadata')

test_requires = ['rasterio', 'dask', 'xarray', 'cartopy>=0.20.0', 'pillow', 'matplotlib', 'scipy', 'zarr',
                 'pytest-lazy-fixtures', 'shapely', 'odc-geo']
extras_require = {'numexpr': ['numexpr'],
                  'quicklook': ['matplotlib', 'cartopy>=0.20.0', 'pillow'],
                  'rasterio': ['rasterio'],
                  'dask': ['dask>=0.16.1'],
                  'cf': ['xarray'],
                  'gradient_search': ['shapely'],
                  'xarray_bilinear': ['xarray', 'dask', 'zarr'],
                  'odc-geo': ['odc-geo'],
                  'tests': test_requires}

all_extras = []
for extra_deps in extras_require.values():
    all_extras.extend(extra_deps)
extras_require['all'] = list(set(all_extras))

if sys.platform.startswith("win"):
    extra_compile_args = []
else:
    extra_compile_args = ["-O3", "-Wno-unused-function"]

extensions = [
    Extension("pyresample.ewa._ll2cr", sources=["pyresample/ewa/_ll2cr.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=extra_compile_args,
              cython_directives={"language_level": 3},
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              ),
    Extension("pyresample.ewa._fornav",
              sources=["pyresample/ewa/_fornav.pyx",
                       "pyresample/ewa/_fornav_templates.cpp"],
              include_dirs=[np.get_include()],
              language="c++", extra_compile_args=extra_compile_args,
              depends=["pyresample/ewa/_fornav_templates.h"],
              cython_directives={"language_level": 3},
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              ),
    Extension("pyresample.gradient._gradient_search",
              sources=["pyresample/gradient/_gradient_search.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=extra_compile_args,
              cython_directives={"language_level": 3},
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              ),
]

cmdclass = versioneer.get_cmdclass(cmdclass={"build_ext": build_ext})

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
          include_package_data=True,
          python_requires='>=3.9',
          install_requires=requirements,
          extras_require=extras_require,
          ext_modules=extensions,
          entry_points=entry_points,
          zip_safe=False,
          classifiers=[
              'Development Status :: 5 - Production/Stable',
              'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
              'Programming Language :: Python',
              'Programming Language :: Cython',
              'Operating System :: OS Independent',
              'Intended Audience :: Science/Research',
              'Topic :: Scientific/Engineering'
          ]
          )
