Reduction of swath data
=======================
Given a swath and a cartesian grid or grid lons and lats, pyresample can reduce the swath data
to only the relevant part covering the grid area. The reduction is coarse in order not to risk removing
relevant data.

From **data_reduce** the function **swath_from_lonlat_grid** can be used to reduce the swath data set to the
area covering the lon lat grid

.. doctest::

 >>> import numpy as np
 >>> from pyresample import geometry, data_reduce
 >>> area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
 ...                                {'a': '6378144.0', 'b': '6356759.0',
 ...                                 'lat_0': '50.00', 'lat_ts': '50.00',
 ...                                 'lon_0': '8.00', 'proj': 'stere'},
 ...                                800, 800,
 ...                                [-1370912.72, -909968.64,
 ...                                 1029087.28, 1490031.36])
 >>> data = np.fromfunction(lambda y, x: y*x, (50, 10))
 >>> lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
 >>> lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
 >>> grid_lons, grid_lats = area_def.get_lonlats()
 >>> reduced_lons, reduced_lats, reduced_data = \
 ... 				data_reduce.swath_from_lonlat_grid(grid_lons, grid_lats,
 ...				lons, lats, data,
 ...				radius_of_influence=3000)

**radius_of_influence** is used to calculate a buffer zone around the grid where swath data points
are not reduced.

The function **get_valid_index_from_lonlat_grid** returns a boolean array of same size as the swath
indicating the relevant swath data points compared to the grid

.. doctest::

 >>> import numpy as np
 >>> from pyresample import geometry, data_reduce
 >>> area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
 ...                                {'a': '6378144.0', 'b': '6356759.0',
 ...                                 'lat_0': '50.00', 'lat_ts': '50.00',
 ...                                 'lon_0': '8.00', 'proj': 'stere'},
 ...                                800, 800,
 ...                                [-1370912.72, -909968.64,
 ...                                 1029087.28, 1490031.36])
 >>> data = np.fromfunction(lambda y, x: y*x, (50, 10))
 >>> lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
 >>> lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
 >>> grid_lons, grid_lats = area_def.get_lonlats()
 >>> valid_index = data_reduce.get_valid_index_from_lonlat_grid(grid_lons, grid_lats,
 ...						lons, lats,
 ...						radius_of_influence=3000)
