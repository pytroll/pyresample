Geometry Utilities
==================

Pyresample provides convenience functions for constructing area
definitions. This includes functions for loading ``AreaDefinition``
from on-disk files. Some of these utility functions are described
below.

AreaDefinition Creation
-----------------------

The main utility function for creating
:class:`~pyresample.geometry.AreaDefinition` objects is the
:func:`~pyresample.utils.create_area_def` function. This function will take
whatever information can be provided to describe a geographic region and
create a valid ``AreaDefinition`` object if possible. If it can't make
a fully specified ``AreaDefinition`` then it will provide a
:class:`~pyresample.geometry.DynamicAreaDefinition` instead. The function
can handle unit conversions and will perform the coordinate calculations
necessary to get an area's ``shape`` and ``area_extent``.

The ``create_area_def`` function has the following required arguments:

* **area_id**: ID of area
* **projection**: Projection parameters as a proj4_dict or proj4_string

and optional arguments:

* **description**: Human-readable description. If not provided, defaults to **area_id**
* **proj_id**: ID of projection (deprecated)
* **units**: Units that provided arguments should be interpreted as. This can be
    one of 'deg', 'degrees', 'meters', 'metres', and any parameter supported by the
    `cs2cs -lu <https://proj4.org/apps/cs2cs.html#cmdoption-cs2cs-lu>`_
    command. Units are determined in the following priority:

    1. units expressed with each variable through a DataArray's attrs attribute.
    2. units passed to ``units``
    3. units used in ``projection``
    4. meters
* **shape**: Number of pixels in the y and x direction following row-column format (height, width)
* **area_extent**: Area extent as a tuple (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
* **upper_left_extent**: x and y coordinates of the upper left corner of the upper left pixel (x, y)
* **center**: x and y coordinate of the center of projection (x, y)
* **resolution**: Size of pixels in the x and y direction (dx, dy)
* **radius**: Length from the center to the left/right and top/bottom outer edges (dx, dy)

.. doctest::

 >>> from pyresample import utils
 >>> area_id = 'ease_sh'
 >>> proj_dict = {'proj': 'laea', 'lat_0': -90, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'}
 >>> center = (0, 0)
 >>> radius = (5326849.0625, 5326849.0625)
 >>> resolution = (25067.525, 25067.525)
 >>> area_def = utils.create_area_def(area_id, proj_dict, center=center, radius=radius, resolution=resolution)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

The ``create_area_def`` function accepts some parameters in multiple forms
to make it as easy as possible. For example, the **resolution** and **radius**
keyword arguments can be specified with one value if ``dx == dy``:

.. doctest::

 >>> proj_string = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
 >>> area_def = utils.create_area_def(area_id, proj_string, center=center,
 ...                              radius=5326849.0625, resolution=25067.525)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

You can also specify parameters in degrees even if the projection space
is defined in meters. For example the below code creates an area in
the mercator projection with radius and resolution defined in degrees.

.. doctest::

 >>> proj_dict = {'proj': 'merc', 'lat_0': 0, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'}
 >>> area_def = utils.create_area_def(area_id, proj_dict, center=(0, 0),
 ...                              radius=(47.90379019311, 43.1355420077),
 ...                              resolution=(0.22542960090875294, 0.22542901929487608),
 ...                              units='degrees', description='Antarctic EASE grid')
 >>> print(area_def)
 Area ID: ease_sh
 Description: Antarctic EASE grid
 Projection: {'a': '6371228.0', 'lat_0': '0.0', 'lon_0': '0.0', 'proj': 'merc', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

If only one of **area_extent** or **shape** can be computed from the
information provided by the user, a
:class:`~pyresample.geometry.DynamicAreaDefinition` object is returned:

.. doctest::

 >>> area_def = utils.create_area_def(area_id, proj_string, radius=radius, resolution=resolution)
 >>> print(type(area_def))
 <class 'pyresample.geometry.DynamicAreaDefinition'>

.. note::

  **radius** and **resolution** are distances, **NOT** coordinates. When expressed as angles,
  they represent the degrees of longitude/latitude away from the center that
  they should span. Hence in these cases **center or area_extent must be provided**.

AreaDefinition Class Methods
----------------------------

There are four class methods available on the
:class:`~pyresample.geometry.AreaDefinition` class utilizing
:func:`~pyresample.utils.create_area_def` providing a simpler interface to the
functionality described in the previous section.
Hence each argument used below is the same as the ``create_area_def`` arguments
described above and can be used in the same way (i.e. units). The following
functions require **area_id** and **projection** along with a few other
arguments:

from_extent
***********

:func:`~pyresample.geometry.AreaDefinition.from_extent`

.. doctest::

 >>> from pyresample import utils
 >>> from pyresample.geometry import AreaDefinition
 >>> area_id = 'ease_sh'
 >>> proj_string = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
 >>> area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
 >>> shape = (425, 425)
 >>> area_def = AreaDefinition.from_extent(area_id, proj_string, shape, area_extent)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

from_circle
***********

:func:`~pyresample.geometry.AreaDefinition.from_circle`

.. doctest::

 >>> proj_dict = {'proj': 'laea', 'lat_0': -90, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'}
 >>> center = (0, 0)
 >>> radius = 5326849.0625
 >>> area_def = AreaDefinition.from_circle(area_id, proj_dict, center, radius, shape=shape)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

.. doctest::

 >>> resolution = 25067.525
 >>> area_def = AreaDefinition.from_circle(area_id, proj_string, center, radius, resolution=resolution)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

from_area_of_interest
*********************

:func:`~pyresample.geometry.AreaDefinition.from_area_of_interest`

.. doctest::

 >>> area_def = AreaDefinition.from_area_of_interest(area_id, proj_dict, shape, center, resolution)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

from_ul_corner
**************

:func:`~pyresample.geometry.AreaDefinition.from_ul_corner`

 >>> upper_left_extent = (-5326849.0625, 5326849.0625)
 >>> area_def = AreaDefinition.from_ul_corner(area_id, proj_string, shape, upper_left_extent, resolution)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

Loading from disk
-----------------

The :func:`~pyresample.area_config.load_area` function can be used to
parse area definitions from a configuration file by giving it the
area file name and regions you wish to load. :func:`~pyresample.area_config.load_area`
takes advantage of :func:`~pyresample.area_config.create_area_def`
and hence allows for the same arguments in the on-disk file.
Pyresample uses the YAML file format to store on-disk area definitions.
Below is an example YAML configuration file showing the various ways
an area might be specified.

.. code-block:: yaml

 boundary:
   area_id: ease_sh
   description: Example of making an area definition using shape and area_extent
   projection:
     proj: laea
     lat_0: -90
     lon_0: 0
     a: 6371228.0
     units: m
   shape: [425, 425]
   area_extent: [-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625]

 boundary_2:
   description: Another example of making an area definition using shape and area_extent
   units: degrees
   projection:
     proj: laea
     lat_0: -90
     lon_0: 0
     a: 6371228.0
     units: m
   shape:
     height: 425
     width: 425
   area_extent:
     lower_left_xy: [-135.0, -17.516001139327766]
     upper_right_xy: [45.0, -17.516001139327766]

 corner:
   description: Example of making an area definition using shape, upper_left_extent, and resolution
   projection:
     proj: laea
     lat_0: -90
     lon_0: 0
     a: 6371228.0
     units: m
   shape: [425, 425]
   upper_left_extent: [-5326849.0625, 5326849.0625]
   resolution: 25067.525

 corner_2:
   area_id: ease_sh
   description: Another example of making an area definition using shape, upper_left_extent, and resolution
   units:  degrees
   projection:
     proj: laea
     lat_0: -90
     lon_0: 0
     a: 6371228.0
     units: m
   shape: [425, 425]
   upper_left_extent:
     x: -45.0
     y: -17.516001139327766
   resolution:
     dx: 25067.525
     dy: 25067.525
     units: meters

 circle:
   description: Example of making an area definition using center, resolution, and radius
   projection:
     proj: laea
     lat_0: -90
     lon_0: 0
     a: 6371228.0
     units: m
   center: [0, 0]
   resolution: [25067.525, 25067.525]
   radius: 5326849.0625

 circle_2:
   area_id: ease_sh
   description: Another example of making an area definition using center, resolution, and radius
   projection:
     proj: laea
     lat_0: -90
     lon_0: 0
     a: 6371228.0
     units: m
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
     units: degrees

 area_of_interest:
   description: Example of making an area definition using shape, center, and resolution
   projection:
     proj: laea
     lat_0: -90
     lon_0: 0
     a: 6371228.0
     units: m
   shape: [425, 425]
   center: [0, 0]
   resolution: [25067.525, 25067.525]

 area_of_interest_2:
   area_id: ease_sh
   description: Another example of making an area definition using shape, center, and resolution
   projection:
     proj: laea
     lat_0: -90
     lon_0: 0
     a: 6371228.0
     units: m
   shape: [425, 425]
   center:
     center: [0, -90]
     units: deg
   resolution:
     resolution: 0.22542974631297721
     units: deg

 epsg:
   area_id: ease_sh
   description: Example of making an area definition using EPSG codes
   projection:
     init: EPSG:3410
   shape: [425, 425]
   area_extent: [-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625]

.. note::

  The `lower_left_xy` and `upper_right_xy` items give the coordinates of the
  outer edges of the corner pixels on the x and y axis respectively. When the
  projection coordinates are longitudes and latitudes, it is expected to
  provide the extent in `longitude, latitude` order.

.. note::

  When using pyproj 2.0+, please use the new ``'EPSG: XXXX'`` syntax
  as the old ``'init: EPSG:XXXX'`` is no longer supported.

If we assume the YAML content is stored in an ``areas.yaml`` file, we can
read a single ``AreaDefinition`` named ``corner`` by doing:

.. doctest::

 >>> from pyresample import load_area
 >>> area_def = load_area('areas.yaml', 'corner')
 >>> print(area_def)
 Area ID: corner
 Description: Example of making an area definition using shape, upper_left_extent, and resolution
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

Several area definitions can be read at once using the region names as a
series of arguments:

.. doctest::

 >>> corner, boundary = load_area('areas.yaml', 'corner', 'boundary')
 >>> print(boundary)
 Area ID: ease_sh
 Description: Example of making an area definition using shape and area_extent
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

Loading from disk (legacy)
--------------------------

For backwards compatibility, we still support the legacy area file format.
Assuming the file **areas.cfg** exists with the following content

.. code-block:: ini

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

 >>> from pyresample import load_area
 >>> area = load_area('areas.cfg', 'ease_nh')
 >>> print(area)
 Area ID: ease_nh
 Description: Arctic EASE grid
 Projection ID: ease_nh
 Projection: {'a': '6371228.0', 'lat_0': '90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

Note: In the configuration file **REGION** maps to **area_id** and **PCS_ID** maps to **proj_id**.

Several area definitions can be read at once using the region names in an argument list:

.. doctest::

 >>> nh_def, sh_def = load_area('areas.cfg', 'ease_nh', 'ease_sh')
 >>> print(sh_def)
 Area ID: ease_sh
 Description: Antarctic EASE grid
 Projection ID: ease_sh
 Projection: {'a': '6371228.0', 'lat_0': '-90.0', 'lon_0': '0.0', 'proj': 'laea', 'units': 'm'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
