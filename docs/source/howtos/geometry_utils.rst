Geometry Utilities
==================

Pyresample provides convenience functions for constructing area
definitions. This includes functions for loading ``AreaDefinition``
from on-disk files, and netCDF/CF files. Some of these utility
functions are described below.

AreaDefinition Creation
-----------------------

The main utility function for creating
:class:`~pyresample.geometry.AreaDefinition` objects is the
:func:`~pyresample.area_config.create_area_def` function. This function will take
whatever information can be provided to describe a geographic region and
create a valid ``AreaDefinition`` object if possible. If it can't make
a fully specified ``AreaDefinition`` then it will provide a
:class:`~pyresample.geometry.DynamicAreaDefinition` instead. The function
can handle unit conversions and will perform the coordinate calculations
necessary to get an area's ``shape`` and ``area_extent``.

The ``create_area_def`` function has the following required arguments:

* **area_id**: ID of area
* **projection**: Projection parameters as a dictionary or string of PROJ
  parameters.

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

 >>> from pyresample import create_area_def
 >>> area_id = 'ease_sh'
 >>> proj_dict = {'proj': 'laea', 'lat_0': -90, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'}
 >>> center = (0, 0)
 >>> radius = (5326849.0625, 5326849.0625)
 >>> resolution = (25067.525, 25067.525)
 >>> area_def = create_area_def(area_id, proj_dict, center=center, radius=radius, resolution=resolution)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'R': '6371228', 'lat_0': '-90', 'lon_0': '0', 'no_defs': 'None', 'proj': 'laea', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

.. note::

    Projection (CRS) information is stored internally using the pyproj
    library's :class:`CRS <pyproj.crs.CRS>` object. To meet certain standards
    for representing CRS information, pyproj may rename parameters or use
    completely different parameters from what you provide.

The ``create_area_def`` function accepts some parameters in multiple forms
to make it as easy as possible. For example, the **resolution** and **radius**
keyword arguments can be specified with one value if ``dx == dy``:

.. doctest::

 >>> proj_string = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
 >>> area_def = create_area_def(area_id, proj_string, center=center,
 ...                              radius=5326849.0625, resolution=25067.525)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'R': '6371228', 'lat_0': '-90', 'lon_0': '0', 'no_defs': 'None', 'proj': 'laea', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

You can also specify parameters in degrees even if the projection space
is defined in meters. For example the below code creates an area in
the mercator projection with radius and resolution defined in degrees.

.. doctest::

 >>> proj_dict = {'proj': 'merc', 'lon_0': 0, 'no_defs': None, 'proj': 'merc', 'R': 6371228, 'k': 1, 'units': 'm'}
 >>> area_def = create_area_def(area_id, proj_dict, center=(0, 0),
 ...                              radius=(47.90379019311, 43.1355420077),
 ...                              resolution=(0.22542960090875294, 0.22542901929487608),
 ...                              units='degrees', description='Antarctic EASE grid')
 >>> print(area_def)
 Area ID: ease_sh
 Description: Antarctic EASE grid
 Projection: {'R': '6371228', 'k': '1', 'lon_0': '0', 'no_defs': 'None', 'proj': 'merc', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

The area definition corresponding to a given lat-lon grid (defined by area extent and resolution)
can be obtained as follows:

.. doctest::

 >>> area_def = create_area_def('my_area',
 ...                            {'proj': 'longlat', 'datum': 'WGS84'},
 ...                            area_extent=[-180, -90, 180, 90],
 ...                            resolution=1,
 ...                            units='degrees',
 ...                            description='Global 1x1 degree lat-lon grid')
 >>> print(area_def)
 Area ID: my_area
 Description: Global 1x1 degree lat-lon grid
 Projection: {'datum': 'WGS84', 'no_defs': 'None', 'proj': 'longlat', 'type': 'crs'}
 Number of columns: 360
 Number of rows: 180
 Area extent: (-180.0, -90.0, 180.0, 90.0)

If only one of **area_extent** or **shape** can be computed from the
information provided by the user, a
:class:`~pyresample.geometry.DynamicAreaDefinition` object is returned:

.. doctest::

 >>> area_def = create_area_def(area_id, proj_string, radius=radius, resolution=resolution)
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
:func:`~pyresample.area_config.create_area_def` providing a simpler interface to the
functionality described in the previous section.
Hence each argument used below is the same as the ``create_area_def`` arguments
described above and can be used in the same way (i.e. units). The following
functions require **area_id** and **projection** along with a few other
arguments:

from_extent
***********

:func:`~pyresample.geometry.AreaDefinition.from_extent`

.. doctest::

 >>> from pyresample.geometry import AreaDefinition
 >>> area_id = 'ease_sh'
 >>> proj_string = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
 >>> area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
 >>> shape = (425, 425)
 >>> area_def = AreaDefinition.from_extent(area_id, proj_string, shape, area_extent)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'R': '6371228', 'lat_0': '-90', 'lon_0': '0', 'no_defs': 'None', 'proj': 'laea', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
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
 Projection: {'R': '6371228', 'lat_0': '-90', 'lon_0': '0', 'no_defs': 'None', 'proj': 'laea', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

.. doctest::

 >>> resolution = 25067.525
 >>> area_def = AreaDefinition.from_circle(area_id, proj_string, center, radius, resolution=resolution)
 >>> print(area_def)
 Area ID: ease_sh
 Description: ease_sh
 Projection: {'R': '6371228', 'lat_0': '-90', 'lon_0': '0', 'no_defs': 'None', 'proj': 'laea', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
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
 Projection: {'R': '6371228', 'lat_0': '-90', 'lon_0': '0', 'no_defs': 'None', 'proj': 'laea', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
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
 Projection: {'R': '6371228', 'lat_0': '-90', 'lon_0': '0', 'no_defs': 'None', 'proj': 'laea', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
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

.. testsetup::

   import os
   import shutil
   # Travis Windows currently doesn't clone symbolic links properly
   # See https://travis-ci.community/t/git-symlinks-support/274
   if os.getenv('TRAVIS_OS_NAME', '') == 'windows':
       os.remove('areas.yaml')
       shutil.copy('../pyresample/test/test_files/areas.yaml', 'areas.yaml')

.. doctest::

 >>> from pyresample import load_area
 >>> import yaml
 >>> area_def = load_area('areas.yaml', 'corner')
 >>> print(area_def)
 Area ID: corner
 Description: Example of making an area definition using shape, upper_left_extent, and resolution
 Projection: {'R': '6371228', 'lat_0': '-90', 'lon_0': '0', 'no_defs': 'None', 'proj': 'laea', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
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
 Projection: {'R': '6371228', 'lat_0': '-90', 'lon_0': '0', 'no_defs': 'None', 'proj': 'laea', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
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
 Projection: {'R': '6371228', 'lat_0': '90', 'lon_0': '0', 'no_defs': 'None', 'proj': 'laea', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
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
 Projection: {'R': '6371228', 'lat_0': '-90', 'lon_0': '0', 'no_defs': 'None', 'proj': 'laea', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

Writing to disk
---------------
To write an area definition to a yaml file to disk use the :func:`~pyresample.geometry.AreaDefinition.dump` function
of the :class:`~pyresample.geometry.AreaDefinition`.

Loading from netCDF/CF
----------------------
``AreaDefinition`` objects can be loaded from netCDF CF_ files with function :func:`~pyresample.utils.cf.load_cf_area`.

>>> from pyresample.utils import load_cf_area

The :func:`~pyresample.utils.cf.load_cf_area` routine offers three call forms:

- Load the ``AreaDefinition`` from a specific CF `grid_mapping` object: with all three of ``variable=``, ``x=``, and ``y=`` ;
- Load the ``AreaDefinition`` sustaining a CF variable: only ``variable=`` ;
- Find and load the valid ``AreaDefinition`` in a CF file: no parameter ;

Consider the following netCDF/CF file: ::

   netcdf cf_nh10km {
   dimensions:
   	xc = 760 ;
   	yc = 1120 ;
   variables:
   	int Polar_Stereographic_Grid ;
   		Polar_Stereographic_Grid:grid_mapping_name = "polar_stereographic" ;
   		Polar_Stereographic_Grid:false_easting = 0. ;
   		Polar_Stereographic_Grid:false_northing = 0. ;
   		Polar_Stereographic_Grid:semi_major_axis = 6378273. ;
   		Polar_Stereographic_Grid:semi_minor_axis = 6356889.44891 ;
   		Polar_Stereographic_Grid:straight_vertical_longitude_from_pole = -45. ;
   		Polar_Stereographic_Grid:latitude_of_projection_origin = 90. ;
   		Polar_Stereographic_Grid:standard_parallel = 70. ;
   	double xc(xc) ;
   		xc:axis = "X" ;
   		xc:units = "km" ;
   		xc:long_name = "x coordinate in Cartesian system" ;
   		xc:standard_name = "projection_x_coordinate" ;
   	double yc(yc) ;
   		yc:axis = "Y" ;
   		yc:units = "km" ;
   		yc:long_name = "y coordinate in Cartesian system" ;
   		yc:standard_name = "projection_y_coordinate" ;
   	float lat(yc, xc) ;
   		lat:long_name = "latitude coordinate" ;
   		lat:standard_name = "latitude" ;
   		lat:units = "degrees_north" ;
   	float lon(yc, xc) ;
   		lon:long_name = "longitude coordinate" ;
   		lon:standard_name = "longitude" ;
   		lon:units = "degrees_east" ;
   	short ice_conc(yc, xc) ;
   		ice_conc:_FillValue = -999s ;
   		ice_conc:grid_mapping = "Polar_Stereographic_Grid" ;
   		ice_conc:coordinates = "lat lon" ;
   		ice_conc:standard_name = "sea_ice_area_fraction" ;
   		ice_conc:units = "%" ;
   		ice_conc:scale_factor = 0.01f ;
   		ice_conc:add_offset = 0.f ;
   		ice_conc:valid_min = 0 ;
   		ice_conc:valid_max = 10000 ;
   // global attributes:
                :Conventions = "CF-1.7"

   }

The three call forms are:


**1st call form:**

>>> area_def, cf_info = load_cf_area('/path/to/cf_nh10km.nc', variable='Polar_Stereographic_Grid', x='xc', y='yc')

This will directly create the AreaDefinition ``area_def`` from the content of the `grid_mapping` variable
'Polar_Stereographic_Grid', and the area extent from the 'xc' and 'yc'.

**2nd call form:**

>>> area_def, cf_info = load_cf_area('/path/to/cf_nh10km.nc', variable='ice_conc')

This will search which `grid_mapping`, `x` and `y` axes sustain the 'ice_conc' variable, and
create the ``AreaDefinition`` from this information.

**3rd call form:**

>>> area_def, cf_info = load_cf_area('/path/to/cf_nh10km.nc')

This will look through the whole netCDF/CF file, and guess all information needed to load a ``AreaDefinition`` object.

.. note::

   The CF convention allows that a single file defines several different `grid_mappings`. At present,
   the 3rd call form of ``load_cf_area()`` will raise a ``ValueError`` exception when this happens.

   If you have several `grid_mappings` in your CF file, be specific which one you want to access with the 1st or 2nd call form.


Although a recommended practice, it cannot be trusted that the 'y' and 'x' axes are in the last two dimensions of a CF variable.
This is because the CF convention does not impose the order of the dimensions of a variable. ``load_cf_area()`` will effectively
look for the variables holding the `x` and `y` coordinates of the Earth mapping projection, not based on the order of the dimensions
of the CF variable.

**Access to additional info from the CF file:**

Not all relevant information can be stored in the ``AreaDefinition`` object. For example, it can be useful to know
what were the names of the variables holding the coordinate variables ('xc' and 'yc' in the example above), or
that the `latitude` and `longitude` associated to the `grid_mapping` are stored in variables 'lat' and 'lon'. Such
information can be useful for writing additional variables to the CF file, or to create a new file that looks
similar to the one we just read.

This information is in a second return value ``cf_info``:

>>> area_def, cf_info = load_cf_area('/path/to/cf_nh10km.nc', with_cf_info=True)

The ``cf_info`` is a ``dict()`` holding additional information about the way the `grid_mapping` information
was coded in the CF file. It may not contain the same amount of information in all three call forms.
For example, the 1st call form does not allow to find
the name of the `latitude` or `longitude` variables, since the 1st call form only gives access to the `grid_mapping`
variable and its coordinate axes.


.. _CF: http://cfconventions.org/cf-conventions/cf-conventions.html


Converting Coordinates
----------------------

The ``AreaDefinition`` have a few handy coordinate conversion methods available:

- :meth:`~pyresample.geometry.AreaDefinition.get_array_coordinates_from_lonlat`
- :meth:`~pyresample.geometry.AreaDefinition.get_array_coordinates_from_projection_coordinates`
- :meth:`~pyresample.geometry.AreaDefinition.get_projection_coordinates_from_lonlat`
- :meth:`~pyresample.geometry.AreaDefinition.get_lonlat_from_array_coordinates`
- :meth:`~pyresample.geometry.AreaDefinition.get_lonlat_from_projection_coordinates`
- :meth:`~pyresample.geometry.AreaDefinition.get_projection_coordinates_from_array_coordinates`

We also have two methods returning integers for array indices:

- :meth:`~pyresample.geometry.AreaDefinition.get_array_indices_from_lonlat`
- :meth:`~pyresample.geometry.AreaDefinition.get_array_indices_from_projection_coordinates`

These two raise a ``ValueError`` if the scalar input coordinates are oustide the extent of the area.
