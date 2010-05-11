Area definitions
================

The cartegraphic definition of areas used by Pyresample is contained in a object of type AreaDefintion. 
The following arguments are needed to initialize an area:

* **area_id** ID of area  
* **name**: Description
* **proj_id**: ID of projection 
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
 >>> name = 'Antarctic EASE grid'
 >>> proj_id = 'ease_sh'
 >>> proj4_args = 'proj=laea, lat_0=-90, lon_0=0, a=6371228.0, units=m'
 >>> x_size = 425
 >>> y_size = 425
 >>> area_extent = (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
 >>> proj_dict = {'a': '6371228.0', 'units': 'm', 'lon_0': '0',
 ...              'proj': 'laea', 'lat_0': '-90'}
 >>> area_def = geometry.AreaDefinition(area_id, name, proj_id, proj_dict, x_size,
 ...                                y_size, area_extent)
 >>> print area_def
 Area ID: ease_sh
 Name: Antarctic EASE grid
 Projection ID: ease_sh
 Projection: {'a': '6371228.0', 'units': 'm', 'lon_0': '0', 'proj': 'laea', 'lat_0': '-90'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

pyresample.utils
----------------
The utils module of pyresample has convenience functions for constructing
area defintions. The function **get_area_def** can construct an area definition
based on a proj4-string or a list of proj4 arguments.

.. doctest::
	
 >>> from pyresample import utils
 >>> area_id = 'ease_sh'
 >>> area_name = 'Antarctic EASE grid'
 >>> proj_id = 'ease_sh'
 >>> proj4_args = '+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m'
 >>> x_size = 425
 >>> y_size = 425
 >>> area_extent = (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
 >>> area_def = utils.get_area_def(area_id, area_name, proj_id, proj4_args, 
 ...                  			   x_size, y_size, area_extent)
 >>> print area_def
 Area ID: ease_sh
 Name: Antarctic EASE grid
 Projection ID: ease_sh
 Projection: {'a': '6371228.0', 'units': 'm', 'lon_0': '0', 'proj': 'laea', 'lat_0': '-90'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)


The **parse_area_file** function can be used to parse area definitions from a configuration file. 
Assuming the file **/tmp/areas.cfg** exists with the following content

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
 >>> areas = utils.parse_area_file('/tmp/areas.cfg', 'ease_nh')
 >>> print areas[0]
 Area ID: ease_nh
 Name: Arctic EASE grid
 Projection ID: ease_nh
 Projection: {'a': '6371228.0', 'units': 'm', 'lon_0': '0', 'proj': 'laea', 'lat_0': '90'}
 Number of columns: 425
 Number of rows: 425
 Area extent: (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)

Note: In the configuration file **REGION** maps to **area_id** and **PCS_ID** maps to **proj_id**. 