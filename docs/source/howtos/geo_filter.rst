Geographic filtering
====================
The module **pyresample.geo_filter** contains classes to filter geo data


GridFilter
----------
Allows for filtering of data based on a geographic mask. The filtering uses a bucket sampling approach.

The following example shows how to select data falling in the upper left and lower right quadrant of
a full globe Plate Carrée projection using an 8x8 filter mask

.. doctest::

 >>> import numpy as np
 >>> from pyresample import geometry, geo_filter
 >>> lons = np.array([-170, -30, 30, 170])
 >>> lats = np.array([20, -40, 50, -80])
 >>> swath_def = geometry.SwathDefinition(lons, lats)
 >>> data = np.array([1, 2, 3, 4])
 >>> filter_area = geometry.AreaDefinition('test', 'test', 'test',
 ...         {'proj' : 'eqc', 'lon_0' : 0.0, 'lat_0' : 0.0},
 ...           8, 8,
 ...          (-20037508.34, -10018754.17, 20037508.34, 10018754.17)
 ...		 )
 >>> filter = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
 ...         [1, 1, 1, 1, 0, 0, 0, 0],
 ...         [1, 1, 1, 1, 0, 0, 0, 0],
 ...         [1, 1, 1, 1, 0, 0, 0, 0],
 ...         [0, 0, 0, 0, 1, 1, 1, 1],
 ...         [0, 0, 0, 0, 1, 1, 1, 1],
 ...         [0, 0, 0, 0, 1, 1, 1, 1],
 ...         [0, 0, 0, 0, 1, 1, 1, 1],
 ...         ])
 >>> grid_filter = geo_filter.GridFilter(filter_area, filter)
 >>> swath_def_filtered, data_filtered = grid_filter.filter(swath_def, data)

Input swath_def and data must match as described in :ref:`swath`.

The returned data will always have a 1D geometry_def and if multiple channels are present the filtered
data will have the shape (number_of_points, channels).
