Pyresample
==========

Pyresample is a python package for resampling geospatial image data. It is the
primary method for resampling in the `SatPy <https://github.com/pytroll/satpy>`_
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
- Bucket resampling (count hits per bin, averaging, ratios)

For nearest neighbor and bilinear interpolation pyresample uses a kd-tree
approach by using the fast KDTree implementation provided by the
`pykdtree <https://github.com/storpipfugl/pykdtree>`_ library.
Pyresample works with numpy arrays and numpy masked arrays. Interfaces to
XArray objects (including dask array support) are provided in separate
Resampler class interfaces and are in active development.
Utility functions are available to easily plot data using Cartopy.

.. versionchanged:: 1.15

    Dropped Python 2 and Python <3.4 support.

Documentation
-------------
.. toctree::
   :maxdepth: 2

   installation
   geo_def
   geometry_utils
   geo_filter
   grid
   swath
   multi
   preproc
   spherical_geometry
   plot
   data_reduce
   roadmap
   API <api/pyresample>


