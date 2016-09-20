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

import imp
# workaround python bug: http://bugs.python.org/issue15881#msg170215
import multiprocessing
import os
import sys

from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

version = imp.load_source('pyresample.version', 'pyresample/version.py')

requirements = ['pyproj', 'numpy', 'configobj']
extras_require = {'pykdtree': ['pykdtree>=1.1.1'],
                  'numexpr': ['numexpr'],
                  'quicklook': ['matplotlib', 'basemap', 'pillow']}

if sys.version_info < (2, 6):
    # multiprocessing is not in the standard library
    requirements.append('multiprocessing')

if sys.platform.startswith("win"):
    extra_compile_args = []
else:
    extra_compile_args = ["-O3", "-Wno-unused-function"]

extensions = [
    Extension("pyresample.ewa._ll2cr", sources=["pyresample/ewa/_ll2cr.pyx"],
              extra_compile_args=extra_compile_args),
    Extension("pyresample.ewa._fornav", sources=["pyresample/ewa/_fornav.pyx", "pyresample/ewa/_fornav_templates.cpp"], language="c++", extra_compile_args=extra_compile_args,
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


class build_ext(_build_ext):
    """Work around to bootstrap numpy includes in to extensions.

    Copied from:

        http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py

    """

    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        set_builtin('__NUMPY_SETUP__', False)
        import numpy
        self.include_dirs.append(numpy.get_include())


if __name__ == "__main__":
    if not os.getenv("USE_CYTHON", False) or cythonize is None:
        print("Cython will not be used. Use environment variable 'USE_CYTHON=True' to use it")

        def cythonize(extensions, **_ignore):
            """Fake function to compile from C/C++ files instead of compiling .pyx files with cython.
            """
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


    setup(name='pyresample',
      version=version.__version__,
      description='Resampling of remote sensing data in Python',
      author='Thomas Lavergne',
      author_email='t.lavergne@met.no',
      package_dir={'pyresample': 'pyresample'},
      packages=find_packages(),
      setup_requires=['numpy'],
      install_requires=requirements,
      extras_require=extras_require,
      cmdclass={'build_ext': build_ext},
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
