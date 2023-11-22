Geometry definitions
====================

The :mod:`pyresample.geometry` module contains classes for describing different
geographic areas using a mesh of points or pixels. Some classes represent
geographic areas made of of evenly spaced/sized pixels, others handle the cases
where the region is described by non-uniform pixels. The best object for describing a
region depends on the use case and the information known about it. The different
classes available in pyresample are described below.

Note that all longitudes and latitudes provided to :mod:`pyresample.geometry`
classes must be in degrees. Additionally, longitudes must be in the
[-180;+180[ validity range.

.. versionchanged:: 1.8.0

    Geometry objects no longer check the validity of the provided longitude
    and latitude coordinates to improve performance. Longitude arrays are
    expected to be between -180 and 180 degrees, latitude -90 to 90 degrees.
    This also applies to all geometry definitions that are provided longitude
    and latitude arrays on initialization. Use
    :func:`~pyresample.utils.check_and_wrap` to preprocess your arrays.

.. _area-definitions:

AreaDefinition
--------------

An :class:`~pyresample.geometry.AreaDefinition`, or ``area``, is the primary
way of specifying a uniformly spaced geographic region in pyresample. It is
also one of the only geometry objects that understands geographic projections.
Areas use the :doc:`PROJ.4 <proj:index>` method for describing projected
coordinate reference systems (CRS). If the projection for an area is not
described by longitude/latitude coordinates then it is typically described
in X/Y coordinates in meters. See the :doc:`PROJ.4 <proj:index>`
documentation for more information on projections and coordinate reference
systems.

The following arguments are needed to initialize an area:

* **area_id**: ID of area
* **description**: Description
* **proj_id**: ID of projection (being deprecated)
* **projection**: Proj4 parameters as a dict or string
* **width**: Number of grid columns
* **height**: Number of grid rows
* **area_extent**: (lower_left_x, lower_left_y, upper_right_x, upper_right_y)

where

* **lower_left_x**: projection x coordinate of lower left corner of lower left pixel
* **lower_left_y**: projection y coordinate of lower left corner of lower left pixel
* **upper_right_x**: projection x coordinate of upper right corner of upper right pixel
* **upper_right_y**: projection y coordinate of upper right corner of upper right pixel

Example:

.. doctest::

 >>> from pyresample.geometry import AreaDefinition
 >>> area_id = 'ease_sh'
 >>> description = 'Antarctic EASE grid'
 >>> proj_id = 'ease_sh'
 >>> projection = {'proj': 'laea', 'lat_0': -90, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'}
 >>> width = 425
 >>> height = 425
 >>> area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
 >>> area_def = AreaDefinition(area_id, description, proj_id, projection,
 ...                           width, height, area_extent)
 >>> area_def
 Area ID: ease_sh
 Description: Antarctic EASE grid
 Projection ID: ease_sh
 Projection: {'R': '6371228', 'lat_0': '-90', 'lon_0': '0', 'no_defs': 'None', 'proj': 'laea', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

You can also specify the projection using a PROJ.4 string

.. doctest::

 >>> projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
 >>> area_def = AreaDefinition(area_id, description, proj_id, projection,
 ...                           width, height, area_extent)

or an `EPSG code <https://www.epsg-registry.org/>`_:

.. doctest::

 >>> projection = '+init=EPSG:3409'  # Use 'EPSG:3409' with pyproj 2.0+
 >>> area_def = AreaDefinition(area_id, description, proj_id, projection,
 ...                           width, height, area_extent)

.. note::

  With pyproj 2.0+ please use the new ``'EPSG:XXXX'`` syntax
  as the old ``'+init=EPSG:XXXX'`` is no longer supported.

Creating an ``AreaDefinition`` can be complex if you don't know everything
about the region being described. Pyresample provides multiple utilities
for creating areas as well as storing them on disk for repeated use. See
the :doc:`geometry_utils` documentation for more information.

GridDefinition
--------------

If the longitude and latitude values for an area are known, the complexity
of an ``AreaDefinition`` can be skipped by using a
:class:`GridDefinition <pyresample.geometry.GridDefinition>` object instead.
Note that although grid definitions are simpler to define they come at the
cost of much higher memory and CPU usage for almost all operations.
The longitude and latitude arrays passed to ``GridDefinition`` are expected to
be evenly spaced. If they are not then a ``SwathDefinition`` should be used
(see below).

.. doctest::

 >>> import numpy as np
 >>> from pyresample.geometry import GridDefinition
 >>> lons = np.ones((100, 100))
 >>> lats = np.ones((100, 100))
 >>> grid_def = GridDefinition(lons=lons, lats=lats)

SwathDefinition
---------------

A swath is defined by the longitude and latitude coordinates for the pixels
it represents. The coordinates represent the center point of each pixel.
Swaths make no assumptions about the uniformity of pixel size and spacing.
This means that operations using then may take longer, but are also accurately
represented.

.. doctest::

 >>> import numpy as np
 >>> from pyresample.geometry import SwathDefinition
 >>> lons = np.ones((500, 20))
 >>> lats = np.ones((500, 20))
 >>> swath_def = SwathDefinition(lons=lons, lats=lats)

Two swaths can be concatenated if their column count matches

.. doctest::

 >>> lons1 = np.ones((500, 20))
 >>> lats1 = np.ones((500, 20))
 >>> swath_def1 = SwathDefinition(lons=lons1, lats=lats1)
 >>> lons2 = np.ones((300, 20))
 >>> lats2 = np.ones((300, 20))
 >>> swath_def2 = SwathDefinition(lons=lons2, lats=lats2)
 >>> swath_def3 = swath_def1.concatenate(swath_def2)

Geographic coordinates and boundaries
-------------------------------------

All geometry definition objects provide access to longitude and latitude
coordinates. The ``get_lonlats()`` method can be used to get
this data and will perform any additional calculations needed to get the
coordinates.

:class:`AreaDefinition <pyresample.geometry.AreaDefinition>` exposes the full
set of projection coordinates as **projection_x_coords** and
**projection_y_coords** properties. Note that for lon/lat projections
(`+proj=latlong`) these coordinates will be in longitude/latitude degrees,
where **projection_x_coords** will be longitude and **projection_y_coords**
will be latitude.

.. versionchanged:: 1.5.1

    Renamed `proj_x_coords` to `projection_x_coords` and `proj_y_coords`
    to `projection_y_coords`.

Get longitude and latitude arrays:

.. doctest::

 >>> area_id = 'ease_sh'
 >>> description = 'Antarctic EASE grid'
 >>> proj_id = 'ease_sh'
 >>> projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
 >>> width = 425
 >>> height = 425
 >>> area_extent = (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
 >>> area_def = AreaDefinition(area_id, description, proj_id, projection,
 ...                           width, height, area_extent)
 >>> lons, lats = area_def.get_lonlats()

Get geocentric X, Y, Z coordinates:

.. doctest::

 >>> area_def = AreaDefinition(area_id, description, proj_id, projection,
 ...                           width, height, area_extent)
 >>> cart_subset = area_def.get_cartesian_coords()[100:200, 350:]

If only the 1D range of a projection coordinate is required it can be extracted
using the **projection_x_coord** or **projection_y_coords** property of a geographic coordinate

.. doctest::

 >>> area_def = AreaDefinition(area_id, description, proj_id, projection,
 ...                           width, height, area_extent)
 >>> proj_x_range = area_def.projection_x_coords
