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
import multiprocessing  # noqa: F401
import versioneer
import os
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

requirements = ['setuptools>=3.2', 'pyproj>=1.9.5.1', 'numpy>=1.10.0', 'configobj',
                'pykdtree>=1.3.1', 'pyyaml', 'six']
extras_require = {'numexpr': ['numexpr'],
                  'quicklook': ['matplotlib', 'cartopy', 'pillow'],
                  'rasterio': ['rasterio'],
                  'dask': ['dask>=0.16.1']}

test_requires = ['rasterio', 'dask', 'xarray', 'cartopy', 'pillow', 'matplotlib', 'scipy']
if sys.version_info < (3, 3):
    test_requires.append('mock')

if sys.platform.startswith("win"):
    extra_compile_args = []
else:
    extra_compile_args = ["-O3", "-Wno-unused-function"]

extensions = [
    Extension("pyresample.ewa._ll2cr", sources=["pyresample/ewa/_ll2cr.pyx"],
              extra_compile_args=extra_compile_args),
    Extension("pyresample.ewa._fornav", sources=["pyresample/ewa/_fornav.pyx",
                                                 "pyresample/ewa/_fornav_templates.cpp"],
              language="c++", extra_compile_args=extra_compile_args,
              depends=["pyresample/ewa/_fornav_templates.h"])
]

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


def set_builtin(name, value):
    if isinstance(__builtins__, dict):
        __builtins__[name] = value
    else:
        setattr(__builtins__, name, value)


cmdclass = versioneer.get_cmdclass()
versioneer_build_ext = cmdclass.get('build_ext', _build_ext)


class build_ext(_build_ext):
    """Work around to bootstrap numpy includes in to extensions.

    Copied from:

        http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py

    """

    def finalize_options(self):
        versioneer_build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        set_builtin('__NUMPY_SETUP__', False)
        import numpy
        self.include_dirs.append(numpy.get_include())


cmdclass['build_ext'] = build_ext

if __name__ == "__main__":
    if not os.getenv("USE_CYTHON", False) or cythonize is None:
        print(
            "Cython will not be used. Use environment variable 'USE_CYTHON=True' to use it")

        def cythonize(extensions, **_ignore):
            """Fake function to compile from C/C++ files instead of compiling .pyx files with cython."""
            for extension in extensions:
                sources = []
                for sfile in extension.sources:
                    path, ext = os.path.splitext(sfile)
                    if ext in ('.pyx', '.py'):
                        if extension.language == 'c++':
                            ext = '.cpp'
                        else:
                            ext = '.c'
                        sfile = path + ext
                    sources.append(sfile)
                extension.sources[:] = sources
            return extensions

    README = open('README.md', 'r').read()
    setup(name='pyresample',
          version=versioneer.get_version(),
          cmdclass=cmdclass,
          description='Geospatial image resampling in Python',
          long_description=README,
          long_description_content_type='text/markdown',
          author='Thomas Lavergne',
          author_email='t.lavergne@met.no',
          package_dir={'pyresample': 'pyresample'},
          packages=find_packages(),
          python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*',
          setup_requires=['numpy'],
          install_requires=requirements,
          extras_require=extras_require,
          tests_require=test_requires,
          ext_modules=cythonize(extensions),
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
