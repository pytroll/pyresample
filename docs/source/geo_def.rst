Geometry definitions
====================
The module `pyresample.geometry <https://pyresample.readthedocs.io/en/latest/API.html#pyresample-geometry>`_
contains classes for describing different kinds of types of remote sensing data geometries.
The use of the different classes is described below.

Remarks
-------

All longitudes and latitudes provided to
`pyresample.geometry <https://pyresample.readthedocs.io/en/latest/API.html#pyresample-geometry>`_ must be in degrees.
Longitudes must additionally be in the [-180;+180[ validity range.

As of version 1.1.1, the
`pyresample.geometry <https://pyresample.readthedocs.io/en/latest/API.html#pyresample-geometry>`_ contructors will
check the range of longitude values, send a warning if some of them fall outside validity range,
and automatically correct the invalid values into [-180;+180[.

Use function `utils.wrap_longitudes <https://pyresample.readthedocs.io/en/latest/API.html#utils.wrap_longitudes>`_
for wrapping longitudes yourself.

AreaDefinition
--------------

The cartographic definition of grid areas used by
`Pyresample <https://pyresample.readthedocs.io/en/latest/API.html#pyresample-api>`_ is contained in an
object of type `AreaDefinition <https://pyresample.readthedocs.io/en/latest/API.html#geometry.AreaDefinition>`_.
The following arguments are needed to initialize an area:

* **area_id** ID of area
* **name**: Description
* **proj_id**: ID of projection (being deprecated)
* **proj_dict**: Proj4 parameters as dict
* **x_size**: Number of grid columns
* **y_size**: Number of grid rows
* **area_extent**: (x_ll, y_ll, x_ur, y_ur)

where

* **x_ll**: projection x coordinate of lower left corner of lower left pixel
* **y_ll**: projection y coordinate of lower left corner of lower left pixel
* **x_ur**: projection x coordinate of upper right corner of upper right pixel
* **y_ur**: projection y coordinate of upper right corner of upper right pixel

Creating an area definition:

.. doctest::

 >>> from pyresample import geometry
 >>> area_id = 'ease_sh'
 >>> description = 'Antarctic EASE grid'
 >>> proj_id = 'ease_sh'
 >>> x_size = 425
 >>> y_size = 425
 >>> area_extent = (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
 >>> proj_dict = {'a': 6371228.0, 'units': 'm', 'lon_0': 0.0,
 ...              'proj': 'laea', 'lat_0': -90.0}
 >>> area_def = geometry.AreaDefinition(area_id, description, proj_id,
 ... 									proj_dict, x_size, y_size, area_extent)
 >>> print(area_def)
 Area ID: ease_sh
 Description: Antarctic EASE grid
 Projection ID: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

pyresample.utils
****************

The `utils <https://pyresample.readthedocs.io/en/latest/API.html#module-utils>`_ module of pyresample
has convenience functions for constructing area definitions. The function
`get_area_def <https://pyresample.readthedocs.io/en/latest/API.html#utils.get_area_def>`_ can
construct an `AreaDefinition <https://pyresample.readthedocs.io/en/latest/API.html#geometry.AreaDefinition>`_
object based on area extent and a proj4-string/dict or a list of proj4 arguments.

.. doctest::

 >>> from pyresample import utils
 >>> area_id = 'ease_sh'
 >>> description = 'Antarctic EASE grid'
 >>> proj_id = 'ease_sh'
 >>> projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
 >>> x_size = 425
 >>> y_size = 425
 >>> area_extent = (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
 >>> area_def = utils.get_area_def(area_id, description, proj_id, projection,
 ...                  			   x_size, y_size, area_extent)
 >>> print(area_def)
 Area ID: ease_sh
 Description: Antarctic EASE grid
 Projection ID: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

The function `from_params <https://pyresample.readthedocs.io/en/latest/API.html#utils.from_params>`_ attempts
to return an `AreaDefinition <https://pyresample.readthedocs.io/en/latest/API.html#geometry.AreaDefinition>`_
object if the number of pixel (shape) and area_extent can be found with the given data below:

Required (positional) arguments:

* **description**: Description
* **projection**: projection parameters as a proj4_dict or proj4_string

Optional (keyword) arguments:

* **area_id**, **proj_id**, and **area_extent**: same as AreaDefinition
* **units**: Default projection units: meters, radians, or degrees
* **shape**: Number of pixels: (height, width)
* **top_left_extent**: The projection x and y coordinates of the upper left corner of the upper left pixel (x_ul, y_ul)
* **center**: Center of projection: (center_x, center_y)
* **pixel_size**: Size of pixels: (x_size, y_size)
* **radius**: Length from the center to the edges of the projection: (x_radius, y_radius)

where

* **center_x** and **center_y**: projection x and y coordinate of the center of projection
* **height** and **width**: number of pixels in y (number of grid rows) and x (number of grid columns) direction
* **x_size** and **y_size**: projection size of pixels in the x and y direction
* **x_radius** and **y_radius**: projection length from the center to the left/right and top/bottom outer edges
* **units** accepts anything with 'm', 'rad', 'deg' or '°'. The order of default is:
    1. units expressed with each variable
    2. units passed to **units**
    3. units used in **projection**
    4. meters
* **shape**, **pixel_size**, and **radius** can be specified with one value when their elements are the same.

.. doctest::

 >>> from pyresample import utils
 >>> from xarray import DataArray
 >>> description = 'Antarctic EASE grid'
 >>> projection = {'a': '6371228.0', 'units': 'm', 'lon_0': '0', 'proj': 'laea', 'lat_0': '-90'}
 >>> area_def = utils.from_params(description, projection, pixel_size=(45, -89.681194),
 ...                              area_extent=(-135.0, -17.516001139327766, 45.0, -17.516001139327766),
 ...                              units='degrees', area_id='ease_sh', proj_id='ease_sh')
 >>> print(area_def)
 Area ID: ease_sh
 Description: Antarctic EASE grid
 Projection ID: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

.. doctest::

 >>> from pyresample import utils
 >>> from xarray import DataArray
 >>> description = 'Antarctic EASE grid'
 >>> projection = {'a': '6371228.0', 'units': 'm', 'lon_0': '0', 'proj': 'laea', 'lat_0': '-90'}
 >>> area_def = utils.from_params(description, projection, pixel_size=25067.525,
 ...                              area_extent=(-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625))
 >>> print(area_def)
 Area ID: Antarctic EASE grid
 Description: Antarctic EASE grid
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

The `load_area <https://pyresample.readthedocs.io/en/latest/API.html#utils.load_area>`_ function can be
used to parse area definitions from a configuration file by giving it the area file name and regions
you wish to load. `load_area <https://pyresample.readthedocs.io/en/latest/API.html#utils.load_area>`_
takes advantage of `from_params <https://pyresample.readthedocs.io/en/latest/API.html#utils.from_params>`_
and hence uses the same arguments.

Assuming the file **areas.yaml** exists with the following content

.. code-block:: yaml

 extents:
  description: extents
  area_id: ease_sh
  proj_id: ease_sh
  projection:
    a: 6371228.0
    units: m
    lon_0: 0
    proj: laea
    lat_0: -90
  shape: [425, 850]
  area_extent: [-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625]

 extents_2:
  description: extents
  area_id: ease_sh
  proj_id: ease_sh
  projection:
    a: 6371228.0
    units: m
    lon_0: 0
    proj: laea
    lat_0: -90
  shape: [425, 850]
  area_extent:
    lower_left_xy: [-5326849.0625, -5326849.0625]
    upper_right_xy: [5326849.0625, 5326849.0625]
    units: m

 geotiff:
   description: geotiff
   area_id: ease_sh
   proj_id: ease_sh
   units: meters
   projection:
     a: 6371228.0
     units: m
     lon_0: 0
     proj: laea
     lat_0: -90
   shape: [425, 850]
   top_left_extent: [-5326849.0625, 5326849.0625]
   pixel_size: [12533.7625, 25067.525]

 circle:
   description: circle
   area_id: ease_sh
   proj_id: ease_sh
   units: meters
   projection:
     a: 6371228.0
     units: m
     lon_0: 0
     proj: laea
     lat_0: -90
   center: [0, 0]
   pixel_size: [12533.7625, 25067.525]
   radius: 5326849.0625

 circle_2:
   description: circle_2
   area_id: ease_sh
   proj_id: ease_sh
   units: meters
   projection:
     a: 6371228.0
     units: m
     lon_0: 0
     proj: laea
     lat_0: -90
   center:
     center_x: 0
     center_y: 0
     units: m
   shape:
     width: 850
     height: 425
   radius:
     x_radius: 5326849.0625
     y_radius: 5326849.0625
     units: m

 area_of_interest:
   description: area_of_interest
   area_id: ease_sh
   proj_id: ease_sh
   units: meters
   projection:
     a: 6371228.0
     units: m
     lon_0: 0
     proj: laea
     lat_0: -90
   shape: [425, 850]
   center:
     center: [0, 0]
     units: m
   pixel_size: [12533.7625, 25067.525]

 area_of_interest_2:
   description: area_of_interest_2
   area_id: ease_sh
   proj_id: ease_sh
   units: meters
   projection:
     a: 6371228.0
     units: m
     lon_0: 0
     proj: laea
     lat_0: -90
   shape:
     shape: [425, 850]
     units: m
   center: [0, 0]
   pixel_size:
     x_size: 12533.7625
     y_size: 25067.525

An area definition dict can be read using

.. doctest::

 >>> from pyresample import utils
 >>> area_def = utils.load_area('areas.yaml', 'geotiff')
 >>> print(area_def)
 Area ID: ease_sh
 Description: geotiff
 Projection ID: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 850
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

.. note::

  The `lower_left_xy` and `upper_right_xy` items give the coordinates of the
  outer edges of the corner pixels on the x and y axis respectively. When the
  projection coordinates are longitudes and latitudes, it is expected to
  provide the extent in `longitude, latitude` order.

Several area definitions can be read at once using the region names in an argument list

.. doctest::

 >>> from pyresample import utils
 >>> geotiff, extents = utils.load_area('areas.yaml', 'geotiff', 'extents')
 >>> print(extents)
 Area ID: ease_sh
 Description: extents
 Projection ID: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 850
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

.. note::

  For backwards compatibility, we still support the legacy area file format:

Assuming the file **areas.cfg** exists with the following content

.. code-block:: bash

 REGION: ease_sh {
	NAME:           Antarctic EASE grid
	PCS_ID:         ease_sh
        PCS_DEF:        proj=laea, lat_0=-90, lon_0=0, a=6371228.0, units=m
        XSIZE:          425
        YSIZE:          425
        AREA_EXTENT:    (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
 };

 REGION: ease_nh {
        NAME:           Arctic EASE grid
        PCS_ID:         ease_nh
        PCS_DEF:        proj=laea, lat_0=90, lon_0=0, a=6371228.0, units=m
        XSIZE:          425
        YSIZE:          425
        AREA_EXTENT:    (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
 };

An area definition dict can be read using

.. doctest::

 >>> from pyresample import utils
 >>> area = utils.load_area('areas.cfg', 'ease_nh')
 >>> print(area)
 Area ID: ease_nh
 Description: Arctic EASE grid
 Projection ID: ease_nh
 Projection: {'a': '6371228.0', 'lat_0': '90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

Note: In the configuration file **REGION** maps to **area_id** and **PCS_ID** maps to **proj_id**.

Several area definitions can be read at once using the region names in an argument list

.. doctest::

 >>> from pyresample import utils
 >>> nh_def, sh_def = utils.load_area('areas.cfg', 'ease_nh', 'ease_sh')
 >>> print(sh_def)
 Area ID: ease_sh
 Description: Antarctic EASE grid
 Projection ID: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

GridDefinition
--------------
If the lons and lats grid values are known, the area definition information can be skipped for some types of
resampling by using a `GridDefinition <https://pyresample.readthedocs.io/en/latest/API.html#geometry.GridDefinition>`_
object instead of an `AreaDefinition <https://pyresample.readthedocs.io/en/latest/API.html#geometry.AreaDefinition>`_
object.

.. doctest::

 >>> import numpy as np
 >>> from pyresample import geometry
 >>> lons = np.ones((100, 100))
 >>> lats = np.ones((100, 100))
 >>> grid_def = geometry.GridDefinition(lons=lons, lats=lats)

SwathDefinition
---------------
A swath is defined by the lon and lat values of the data points

.. doctest::

 >>> import numpy as np
 >>> from pyresample import geometry
 >>> lons = np.ones((500, 20))
 >>> lats = np.ones((500, 20))
 >>> swath_def = geometry.SwathDefinition(lons=lons, lats=lats)

Two swaths can be concatenated if their column count matches

.. doctest::

 >>> import numpy as np
 >>> from pyresample import geometry
 >>> lons1 = np.ones((500, 20))
 >>> lats1 = np.ones((500, 20))
 >>> swath_def1 = geometry.SwathDefinition(lons=lons1, lats=lats1)
 >>> lons2 = np.ones((300, 20))
 >>> lats2 = np.ones((300, 20))
 >>> swath_def2 = geometry.SwathDefinition(lons=lons2, lats=lats2)
 >>> swath_def3 = swath_def1.concatenate(swath_def2)

Geographic coordinates and boundaries
-------------------------------------
A ***definition** object allows for retrieval of geographic coordinates using array slicing
(slice stepping is currently not supported).

All ***definition** objects expose the coordinates **lons**, **lats** and **cartesian_coords**.
`AreaDefinition <https://pyresample.readthedocs.io/en/latest/API.html#geometry.AreaDefinition>`_ exposes the
full set of projection coordinates as **projection_x_coords** and **projection_y_coords**. Note that in the
case of projection coordinates expressed in longitude and latitude, **projection_x_coords** will be longitude
and **projection_y_coords** will be latitude.

.. versionchanged:: 1.5.1

    Renamed `proj_x_coords` to `projection_x_coords` and `proj_y_coords`
    to `projection_y_coords`.

Get full coordinate set:

.. doctest::

 >>> from pyresample import utils
 >>> area_id = 'ease_sh'
 >>> description = 'Antarctic EASE grid'
 >>> proj_id = 'ease_sh'
 >>> projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
 >>> x_size = 425
 >>> y_size = 425
 >>> area_extent = (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
 >>> area_def = utils.get_area_def(area_id, description, proj_id, projection,
 ...                               x_size, y_size, area_extent)
 >>> lons, lats = area_def.get_lonlats()

Get slice of coordinate set:

.. doctest::

 >>> from pyresample import utils
 >>> area_id = 'ease_sh'
 >>> description = 'Antarctic EASE grid'
 >>> proj_id = 'ease_sh'
 >>> projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
 >>> x_size = 425
 >>> y_size = 425
 >>> area_extent = (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
 >>> area_def = utils.get_area_def(area_id, description, proj_id, projection,
 ...                               x_size, y_size, area_extent)
 >>> cart_subset = area_def.get_cartesian_coords()[100:200, 350:]

If only the 1D range of a projection coordinate is required it can be extracted
using the **projection_x_coord** or **projection_y_coords** property of a geographic coordinate

.. doctest::

 >>> from pyresample import utils
 >>> area_id = 'ease_sh'
 >>> description = 'Antarctic EASE grid'
 >>> proj_id = 'ease_sh'
 >>> projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
 >>> x_size = 425
 >>> y_size = 425
 >>> area_extent = (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
 >>> area_def = utils.get_area_def(area_id, description, proj_id, projection,
 ...                  			   x_size, y_size, area_extent)
 >>> proj_x_range = area_def.projection_x_coords

Spherical geometry operations
-----------------------------
Some basic spherical operations are available for ***definition** objects. The
spherical geometry operations are calculated based on the corners of a
GeometryDefinition (`GridDefinition <https://pyresample.readthedocs.io/en/latest/API.html#geometry.GridDefinition>`_,
`AreaDefinition <https://pyresample.readthedocs.io/en/latest/API.html#geometry.AreaDefinition>`_, or
2D `SwathDefinition <https://pyresample.readthedocs.io/en/latest/API.html#geometry.SwathDefinition>`_) and assuming the
edges are great circle arcs.

It can be tested if geometries overlaps

.. doctest::

 >>> import numpy as np
 >>> from pyresample import utils
 >>> area_id = 'ease_sh'
 >>> description = 'Antarctic EASE grid'
 >>> proj_id = 'ease_sh'
 >>> projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
 >>> x_size = 425
 >>> y_size = 425
 >>> area_extent = (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
 >>> area_def = utils.get_area_def(area_id, description, proj_id, projection,
 ...                  			   x_size, y_size, area_extent)
 >>> lons = np.array([[-40, -11.1], [9.5, 19.4], [65.5, 47.5], [90.3, 72.3]])
 >>> lats = np.array([[-70.1, -58.3], [-78.8, -63.4], [-73, -57.6], [-59.5, -50]])
 >>> swath_def = geometry.SwathDefinition(lons, lats)
 >>> print(swath_def.overlaps(area_def))
 True

The fraction of overlap can be calculated

.. doctest::

 >>> import numpy as np
 >>> from pyresample import utils
 >>> area_id = 'ease_sh'
 >>> description = 'Antarctic EASE grid'
 >>> proj_id = 'ease_sh'
 >>> projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
 >>> x_size = 425
 >>> y_size = 425
 >>> area_extent = (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
 >>> area_def = utils.get_area_def(area_id, description, proj_id, projection,
 ...                  			   x_size, y_size, area_extent)
 >>> lons = np.array([[-40, -11.1], [9.5, 19.4], [65.5, 47.5], [90.3, 72.3]])
 >>> lats = np.array([[-70.1, -58.3], [-78.8, -63.4], [-73, -57.6], [-59.5, -50]])
 >>> swath_def = geometry.SwathDefinition(lons, lats)
 >>> overlap_fraction = swath_def.overlap_rate(area_def)

And the polygon defining the (great circle) boundaries over the overlapping area can be calculated

.. doctest::

 >>> import numpy as np
 >>> from pyresample import utils
 >>> area_id = 'ease_sh'
 >>> description = 'Antarctic EASE grid'
 >>> proj_id = 'ease_sh'
 >>> projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
 >>> x_size = 425
 >>> y_size = 425
 >>> area_extent = (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
 >>> area_def = utils.get_area_def(area_id, description, proj_id, projection,
 ...                  			   x_size, y_size, area_extent)
 >>> lons = np.array([[-40, -11.1], [9.5, 19.4], [65.5, 47.5], [90.3, 72.3]])
 >>> lats = np.array([[-70.1, -58.3], [-78.8, -63.4], [-73, -57.6], [-59.5, -50]])
 >>> swath_def = geometry.SwathDefinition(lons, lats)
 >>> overlap_polygon = swath_def.intersection(area_def)

It can be tested if a (lon, lat) point is inside a GeometryDefinition

.. doctest::

 >>> import numpy as np
 >>> from pyresample import utils
 >>> area_id = 'ease_sh'
 >>> description = 'Antarctic EASE grid'
 >>> proj_id = 'ease_sh'
 >>> projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
 >>> x_size = 425
 >>> y_size = 425
 >>> area_extent = (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
 >>> area_def = utils.get_area_def(area_id, description, proj_id, projection,
 ...                  			   x_size, y_size, area_extent)
 >>> print((0, -90) in area_def)
 True
