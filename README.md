[![Build Status](https://github.com/pytroll/pyresample/workflows/CI/badge.svg?branch=main)](https://github.com/pytroll/pyresample/actions?query=workflow%3A%22CI%22)
[![Coverage Status](https://coveralls.io/repos/github/pytroll/pyresample/badge.svg?branch=main)](https://coveralls.io/github/pytroll/pyresample?branch=main)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3372769.svg)](https://doi.org/10.5281/zenodo.3372769)


Pyresample
----------

Pyresample is a python package for resampling geospatial image data. It is the
primary method for resampling in the [Satpy](https://github.com/pytroll/satpy)
library, but can also be used as a standalone library. Resampling or
reprojection is the process of mapping input geolocated data points to a
new target geographic projection and area.

Pyresample can operate on both fixed grids of data and geolocated swath data.
To describe these data Pyresample uses various "geometry" objects including
the `AreaDefinition` and `SwathDefinition` classes.

Pyresample offers multiple resampling algorithms including:

- Nearest Neighbor
- Elliptical Weighted Average (EWA)
- Bilinear

For nearest neighbor and bilinear interpolation pyresample uses a kd-tree
approach by using the fast KDTree implementation provided by the
[pykdtree](https://github.com/storpipfugl/pykdtree) library.
Pyresample works with numpy arrays and numpy masked arrays. Interfaces to
XArray objects (including dask array support) are provided in separate
Resampler class interfaces and are in active development.
Utility functions are available to easily plot data using Cartopy.

[Documentation](https://pyresample.readthedocs.org/en/latest/)

See [pytroll.github.io](http://pytroll.github.io/) for more information on the
PyTroll group and related packages.

Citation
----------
Hoese, D., Raspaud, M., Lahtinen, P., Roberts, W., Lavergne, et al. (2020). pytroll/pyresample: Version 1.16.0. Zenodo. [https://doi.org/10.5281/zenodo.3372769](https://doi.org/10.5281/zenodo.3372769)
