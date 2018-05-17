[![Build Status](https://travis-ci.org/pytroll/pyresample.svg?branch=master)](https://travis-ci.org/pytroll/pyresample)
[![Build status](https://ci.appveyor.com/api/projects/status/a34o4utf8dqjsob1/branch/master?svg=true)](https://ci.appveyor.com/project/pytroll/pyresample/branch/master)
[![codebeat badge](https://codebeat.co/badges/2b9f14bc-758c-4fe1-967d-85b11e934983)](https://codebeat.co/projects/github-com-pytroll-pyresample-master)

Python package for geospatial resampling
----------------------------------------

Resampling (reprojection) of geospatial image data in Python.
Pyresample uses a kd-tree approach for resampling.
Pyresample is designed for resampling of remote sensing data and supports resampling from both fixed grids and geolocated swath data.
Several types of resampling are supported including nearest neighbour, gaussian weighting and weighting with a user defined radial function.
Pyresample works with Numpy arrays including support for masked arrays.
Support for parallel resampling using multiple processor cores.
Plotting capablity using Basemap. As of v0.8.0 [pykdtree](https://github.com/storpipfugl/pykdtree) can be used to speed up processing.

Pyresample is tested with Python 2.7 and 3.6, but should additionally work
on Python 3.4+.

[Documentation](https://pyresample.readthedocs.org/en/latest/)
Look at [pytroll.org](http://pytroll.org/) for more information.
