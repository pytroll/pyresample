Geometry definitions
====================
The module :mod:`pyresample.geometry <geometry>` contains classes for describing different kinds
of types of remote sensing data geometries. The use of the different classes is described below.

Remarks
-------

All longitudes and latitudes provided to :mod:`pyresample.geometry <geometry>` must be
in degrees. Longitudes must additionally be in the [-180;+180[ validity range.

As of version 1.1.1, the :mod:`pyresample.geometry <geometry>` contructors will
check the range of longitude values, send a warning if some of them fall outside validity range,
and automatically correct the invalid values into [-180;+180[.

Use function :mod:`utils.wrap_longitudes <utils.wrap_longitudes>` for wrapping longitudes yourself.

AreaDefinition
--------------

The cartographic definition of grid areas used by Pyresample is
contained in an object of type :mod:`AreaDefinition <geometry.AreaDefinition>`
The following arguments are needed to initialize an area:

* **area_id** ID of area
* **name**: Description
* **proj_id**: ID of projection (being deprecated)
* **proj_dict**: Proj4 parameters as dict
* **width**: Number of grid columns
* **height**: Number of grid rows
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
 >>> proj_dict = {'a': '6371228.0', 'units': 'm', 'lon_0': '0', 'proj': 'laea', 'lat_0': '-90'}
 >>> width = 425
 >>> height = 425
 >>> area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
 >>> area_def = geometry.AreaDefinition(area_id, description, proj_id, proj_dict,
 ...                                    width, height, area_extent)
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

The :mod:`utils <utils>` module of Pyresample
has convenience functions for constructing area definitions. The function
:mod:`get_area_def <utils.get_area_def>` can construct an
:mod:`AreaDefinition <geometry.AreaDefinition>` object based on
area_extent and a proj4-string/dict or a list of proj4 arguments.

.. doctest::

 >>> from pyresample import utils
 >>> area_id = 'ease_sh'
 >>> description = 'Antarctic EASE grid'
 >>> proj_id = 'ease_sh'
 >>> proj_string = '+a=6371228.0 +units=m +lon_0=0 +proj=laea +lat_0=-90'
 >>> width = 425
 >>> height = 425
 >>> area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
 >>> area_def = utils.get_area_def(area_id, description, proj_id, proj_string,
 ...                               width, height, area_extent)
 >>> print(area_def)
 Area ID: ease_sh
 Description: Antarctic EASE grid
 Projection ID: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

The function :mod:`from_params <utils.from_params>` attempts to return
an :mod:`AreaDefinition <geometry.AreaDefinition>` object if the number
of pixels (shape) and area_extent can be found with the given data below:

Required arguments:

* **area_id**: ID of area
* **projection**: Projection parameters as a proj4_dict or proj4_string

Optional arguments:

* **description**: Description. If not provided, defaults to **area_id**
* **proj_id**: ID of projection (being deprecated)
* **units**: Default projection units: meters, radians, or degrees. Defaults to: units used in **projection**, meters.
* **area_extent**: Area extent as a list (x_ll, y_ll, x_ur, y_ur)
* **shape**: Number of pixels in the y (grid rows) and x (grid columns) direction (height, width)
* **top_left_extent**: Projection x and y coordinates of the upper left corner of the upper left pixel (x, y)
* **center**: Projection x and y coordinate of the center of projection (x, y)
* **resolution**: Projection size of pixels in the x and y direction (dx, dy)
* **radius**: Projection length from the center to the left/right and top/bottom outer edges (dx, dy)

.. doctest::

 >>> from pyresample import utils
 >>> area_id = 'ease_sh'
 >>> proj_dict = {'a': '6371228.0', 'units': 'm', 'lon_0': '0', 'proj': 'laea', 'lat_0': '-90'}
 >>> center = (0, 0)
 >>> radius = (5326849.0625, 5326849.0625)
 >>> resolution = (25067.525, 25067.525)
 >>> area_def = utils.from_params(area_id, proj_dict, center=center,
 ...                              radius=radius, resolution=resolution)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

When **radius**'s or **resolution**'s elements are the same, they can be passed as a single number:

.. doctest::

 >>> proj_string = '+a=6371228.0 +units=m +lon_0=0 +proj=laea +lat_0=-90'
 >>> area_def = utils.from_params(area_id, proj_string, center=center,
 ...                              radius=5326849.0625, resolution=25067.525)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

An example with degrees as units using a mercator projection:

.. doctest::

 >>> proj_dict = {'a': '6371228.0', 'units': 'm', 'lon_0': '0', 'proj': 'merc', 'lat_0': '-90'}
 >>> area_def = utils.from_params(area_id, proj_dict, center=(0, 0),
 ...                              radius=(47.90379019311, 43.1355420077),
 ...                              resolution=0.225429746313, units='degrees',
 ...                              description='Antarctic EASE grid')
 >>> print(area_def)
 Area ID: ease_sh
 Description: Antarctic EASE grid
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'merc', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

.. note::

  **radius** and **pixel size** are distances, **NOT** coordinates. When expressed as angles,
  they represent the degrees or radians of longitude/latitude away from the center that
  they should span. Hence in these cases **center must be provided or findable**.

There are four subfunctions of :mod:`AreaDefinition <geometry.AreaDefinition>` utilizing
:mod:`from_params <utils.from_params>` to guarantee that an area definition is made. Hence
each argument below is the same as above and can take the same optional arguments as
:mod:`from_params <utils.from_params>` (i.e. units). The following functions require an
**area_id** and **projection** along with a few other arguments:

:mod:`from_extent <geometry.AreaDefinition.from_extent>`:

.. doctest::

 >>> from pyresample import utils
 >>> area_id = 'ease_sh'
 >>> proj_string = '+a=6371228.0 +units=m +lon_0=0 +proj=laea +lat_0=-90'
 >>> area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
 >>> shape = (425, 425)
 >>> area_def = geometry.AreaDefinition.from_extent(area_id, proj_string,
 ...                                                area_extent, shape)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

:mod:`from_circle <geometry.AreaDefinition.from_circle>`

.. doctest::

 >>> proj_dict = {'a': '6371228.0', 'units': 'm', 'lon_0': '0', 'proj': 'laea', 'lat_0': '-90'}
 >>> center = (0, 0)
 >>> radius = 5326849.0625
 >>> area_def = geometry.AreaDefinition.from_circle(area_id, proj_dict, center,
 ...                                                radius, shape=shape)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

.. doctest::

 >>> resolution = 25067.525
 >>> area_def = geometry.AreaDefinition.from_circle(area_id, proj_string, center,
 ...                                                radius, resolution=resolution)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

:mod:`from_area_of_interest <geometry.AreaDefinition.from_area_of_interest>`

.. doctest::

 >>> area_def = geometry.AreaDefinition.from_area_of_interest(area_id, proj_dict, center,
 ...                                                          resolution, shape)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

:mod:`from_geotiff <geometry.AreaDefinition.from_geotiff>`

 >>> top_left_extent = (-5326849.0625, 5326849.0625)
 >>> area_def = geometry.AreaDefinition.from_geotiff(area_id, proj_string, top_left_extent,
 ...                                                 resolution, shape)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

The :mod:`load_area <utils.load_area>` function can be used to
parse area definitions from a configuration file by giving it the
area file name and regions you wish to load. :mod:`load_area <utils.load_area>`
takes advantage of :mod:`from_params <utils.from_params>`
and hence uses the same arguments.

Assuming the file **areas.yaml** exists with the following content

.. code-block:: yaml

 boundary:
   area_id: ease_sh
   description: Example of finding an area definition using shape and area_extent
   projection:
     a: 6371228.0
     units: m
     lon_0: 0
     proj: laea
     lat_0: -90
   shape: [425, 425]
   area_extent: [-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625]

 boundary_2:
   description: Another example of finding an area definition using shape and area_extent
   units: degrees
   projection:
     a: 6371228.0
     units: m
     lon_0: 0
     proj: laea
     lat_0: -90
   shape:
     height: 425
     width: 425
   area_extent:
     lower_left_xy: [-135.0, -17.516001139327766]
     upper_right_xy: [45.0, -17.516001139327766]

 corner:
   description: Example of finding an area definition using shape, top_left_extent, and resolution
   projection:
     a: 6371228.0
     units: m
     lon_0: 0
     proj: laea
     lat_0: -90
   shape: [425, 425]
   top_left_extent: [-5326849.0625, 5326849.0625]
   resolution: 25067.525

 corner_2:
   area_id: ease_sh
   description: Another example of finding an area definition using shape, top_left_extent, and resolution
   units:  °
   projection:
     a: 6371228.0
     units: m
     lon_0: 0
     proj: laea
     lat_0: -90
   shape: [425, 425]
   top_left_extent:
     x: -45.0
     y: -17.516001139327766
   resolution:
     dx: 25067.525
     dy: 25067.525
     units: meters

 circle:
   description: Example of finding an area definition using center, resolution, and radius
   projection:
     a: 6371228.0
     units: m
     lon_0: 0
     proj: laea
     lat_0: -90
   center: [0, 0]
   resolution: [25067.525, 25067.525]
   radius: 5326849.0625

 circle_2:
   area_id: ease_sh
   description: Another example of finding an area definition using center, resolution, and radius
   projection:
     a: 6371228.0
     units: m
     lon_0: 0
     proj: laea
     lat_0: -90
   center:
     x: 0
     y: -90
     units: degrees
   shape:
     width: 425
     height: 425
   radius:
     dx: 49.4217406986
     dy: 49.4217406986
     units: °

 area_of_interest:
   description: Example of finding an area definition using shape, center, and resolution
   projection:
     a: 6371228.0
     units: m
     lon_0: 0
     proj: laea
     lat_0: -90
   shape: [425, 425]
   center: [0, 0]
   resolution: [25067.525, 25067.525]

 area_of_interest_2:
   area_id: ease_sh
   description: Another example of finding an area definition using shape, center, and resolution
   projection:
     a: 6371228.0
     units: m
     lon_0: 0
     proj: laea
     lat_0: -90
   shape: [425, 425]
   center:
     center: [0, -1.570796]
     units: radians
   resolution:
     resolution: 0.0039344913
     units: radians

.. note::

  The `lower_left_xy` and `upper_right_xy` items give the coordinates of the
  outer edges of the corner pixels on the x and y axis respectively. When the
  projection coordinates are longitudes and latitudes, it is expected to
  provide the extent in `longitude, latitude` order.

An area definition dict can be read using

.. doctest::

 >>> from pyresample import utils
 >>> area_def = utils.load_area('areas.yaml', 'corner')
 >>> print(area_def)
 Area ID: corner
 Description: Example of finding an area definition using shape, top_left_extent, and resolution
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

Several area definitions can be read at once using the region names in an argument list

.. doctest::

 >>> corner, boundary = utils.load_area('areas.yaml', 'corner', 'boundary')
 >>> print(boundary)
 Area ID: ease_sh
 Description: Example of finding an area definition using shape and area_extent
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
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
If the lons and lats grid values are known, the area definition information can be skipped for
some types of resampling by using a :mod:`GridDefinition <geometry.GridDefinition>`
object instead of an :mod:`AreaDefinition <geometry.AreaDefinition>` object.

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
:mod:`AreaDefinition <geometry.AreaDefinition>` exposes the full set of projection coordinates
as **projection_x_coords** and **projection_y_coords**. Note that in the case of projection
coordinates expressed in longitude and latitude, **projection_x_coords** will be longitude
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
 >>> width = 425
 >>> height = 425
 >>> area_extent = (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
 >>> area_def = utils.get_area_def(area_id, description, proj_id, projection,
 ...                               width, height, area_extent)
 >>> lons, lats = area_def.get_lonlats()

Get slice of coordinate set:

.. doctest::

 >>> area_def = utils.get_area_def(area_id, description, proj_id, projection,
 ...                               width, height, area_extent)
 >>> cart_subset = area_def.get_cartesian_coords()[100:200, 350:]

If only the 1D range of a projection coordinate is required it can be extracted
using the **projection_x_coord** or **projection_y_coords** property of a geographic coordinate

.. doctest::

 >>> area_def = utils.get_area_def(area_id, description, proj_id, projection,
 ...                  			   width, height, area_extent)
 >>> proj_x_range = area_def.projection_x_coords

Spherical geometry operations
-----------------------------
Some basic spherical operations are available for ***definition** objects. The
spherical geometry operations are calculated based on the corners of a GeometryDefinition
(:mod:`GridDefinition <geometry.GridDefinition>`, :mod:`AreaDefinition <geometry.AreaDefinition>`, or a 2D
:mod:`SwathDefinition <geometry.SwathDefinition>`) and assuming the edges are great circle arcs.

It can be tested if geometries overlaps

.. doctest::

 >>> import numpy as np
 >>> from pyresample import utils
 >>> area_id = 'ease_sh'
 >>> description = 'Antarctic EASE grid'
 >>> proj_id = 'ease_sh'
 >>> projection = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
 >>> width = 425
 >>> height = 425
 >>> area_extent = (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
 >>> area_def = utils.get_area_def(area_id, description, proj_id, projection,
 ...                  			   width, height, area_extent)
 >>> lons = np.array([[-40, -11.1], [9.5, 19.4], [65.5, 47.5], [90.3, 72.3]])
 >>> lats = np.array([[-70.1, -58.3], [-78.8, -63.4], [-73, -57.6], [-59.5, -50]])
 >>> swath_def = geometry.SwathDefinition(lons, lats)
 >>> print(swath_def.overlaps(area_def))
 True

The fraction of overlap can be calculated

.. doctest::

 >>> overlap_fraction = swath_def.overlap_rate(area_def)
 >>> print(overlap_fraction)
 0.05843953132633209

And the polygon defining the (great circle) boundaries over the overlapping area can be calculated

.. doctest::

 >>> overlap_polygon = swath_def.intersection(area_def)
 >>> print(overlap_polygon)
 [(-40.0, -70.1), (-11.1, -58.3), (72.3, -50.0), (90.3, -59.5)]

It can be tested if a (lon, lat) point is inside a GeometryDefinition

.. doctest::

 >>> print((0, -90) in area_def)
 True
