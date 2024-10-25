Pyresample
==========

Pyresample is a python package for resampling geospatial image data. It is the
primary method for resampling in the `SatPy <https://github.com/pytroll/satpy>`_
library, but can also be used as a standalone library.

You can use pyresample to transform data in one coordinate system to another
coordinate system (ex. mercator projection) using one of the available
resampling algorithms. For more information on how
to resample your data see our :doc:`howtos/index` section. If resampling is a
new concept for you then you may find the :doc:`concepts/index` section helpful
in explaining resampling without code or see the :doc:`tutorials/index` to
walk you through the process with some artificial data.

To be able to resample data, Pyresample must have a good understanding of the
geometry of the area these pixels represent. Data to be resampled can consist
of uniformly spaced/gridded pixels or a variably spaced "swath" of pixels.
Pyresample uses "geometry" objects to describe the different
properties of these geolocated datasets. If concepts like projections,
pixel resolutions, or representing data on a sphere or spheroid are new to you
it is recommended you start with the :doc:`concepts/index` section to learn
more without worrying too much about the actual code. After that, the
:doc:`tutorials/index` and :doc:`howtos/index` sections will be able to
show you the code needed to apply these concepts to your data.

Throughout the documentation you'll find information on various utilities
provided by Pyresample to accomplish things like making plots with Cartopy
or describing data from a NetCDF or GeoTIFF file. Pyresample is generally
able to handle data represented as numpy arrays, numpy masked arrays, and
in some parts Xarray DataArray objects and dask arrays. In addition to these
libraries Pyresample also benefits from the works of the
`pykdtree <https://github.com/storpipfugl/pykdtree>`_ and
`shapely <https://shapely.readthedocs.io/en/stable/>`_ libraries. Pyresample
includes Python extension code written in `Cython <https://cython.org/>`_
in the more performance critical portions of the library.

.. warning::

   This documentation is still actively being redesigned and rewritten. Some
   information isn't where you might expect it or might not meet your
   expectations based on the quality and approach of other pieces of
   documentation. Feel free to leave feedback as a GitHub issue, but know
   that we're working on making this documentation and the library better.

Getting Help
------------

Having trouble installing or using Pyresample? Feel free to ask questions at
any of the contact methods for the Pytroll group
`here <https://pytroll.github.io/#getting-in-touch>`_ or file an issue on
`Pyresample's GitHub page <https://github.com/pytroll/pyresample/issues>`_.

Documentation
-------------

.. toctree::
   :maxdepth: 2

   concepts/index
   tutorials/index
   howtos/index
   reference
