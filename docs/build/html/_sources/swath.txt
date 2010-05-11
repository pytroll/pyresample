Resampling of swath data
========================

Pyresample can be used to resample a swath dataset to a grid. Resampling can be done using nearest neighbour method, Guassian weighting, weighting with an arbitrary radial function or bucket resampling.

pyresample.swath
----------------

This module contains several functions for resampling swath data.

Note distance calculation is approximated with cartesian distance.

Masked arrays can be used as data input. In order to have undefined pixels masked out instead of 
assigned a fill value set **fill_value=None** when calling the **resample_*** function.

resample_nearest
****************

Function for resampling using nearest neighbour method.

Example showing how to resample a generated swath dataset to a grid using nearest neighbour method:

.. doctest::

 >>> import numpy as np
 >>> from pyresample import swath, geometry
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
 >>> result = swath.resample_nearest(lons.ravel(), lats.ravel(), data.ravel(),
 ... area_def, radius_of_influence=50000, epsilon=100)

Lons, lats and data arguments are numpy arrays. Lons and lats are 1D arrays and data is either a 1D array or an array of shape (swath_size, channels).

Note the keyword arguments:

* **radius_of_influence**: The radius around each grid pixel in meters to search for neighbours in the swath.
* **epsilon**: Allowed uncertainty in nearest neighbour search in meters. Allowing for uncertanty decreases execution time.

If **data** if a masked array the mask will follow the neighbour pixel assignment.

resample_gauss
**************

Function for resampling using nearest Gussian weighting. The Gauss weigh function is defined as exp(-dist^2/sigma^2).

Example showing how to resample a generated swath dataset to a grid using Gaussian weighting:

.. doctest::

 >>> import numpy as np
 >>> from pyresample import swath, geometry
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
 >>> result = swath.resample_gauss(lons.ravel(), lats.ravel(), data.ravel(), 
 ... area_def, radius_of_influence=50000, sigmas=25000)

If more channels are present in **data** the keyword argument **sigmas** must be a list containing a sigma for each channel.

If **data** if a masked array any pixel in the result data that has been "contaminated" by weighting of a masked pixel is masked.

resample_custom
***************

Function for resampling using arbitrary radial weight functions.

Example showing how to resample a generated swath dataset to a grid using an arbitrary radial weight function:

.. doctest::

 >>> import numpy as np
 >>> from pyresample import swath, geometry 
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
 >>> wf = lambda r: 1 - r/100000.0
 >>> result  = swath.resample_custom(lons.ravel(), lats.ravel(), data.ravel(),
 ...  area_def, radius_of_influence=50000, weight_funcs=wf)

If more channels are present in **data** the keyword argument **weight_funcs** must be a list containing a radial function for each channel.

If **data** if a masked array any pixel in the result data that has been "contaminated" by weighting of a masked pixel is masked.

Resampling from neighbour info
******************************
The resampling can be split in two steps: 

First get arrays containing information about the nearest neighbours to each grid point. 
Then use these arrays to retrive the resampling result.

This approch can be useful if several datasets based on the same swath are to be resampled. The computational 
heavy task of calculating the neighbour information can be done once and the result can be used to 
retrieve the resampled data from each of the datasets fast.

.. doctest::

 >>> import numpy as np
 >>> from pyresample import swath, geometry
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
 >>> valid_index, index_array, distance_array = \
 ...                        swath.get_neighbour_info(lons.ravel(), lats.ravel(), 
 ...                                                 area_def, 50000,  
 ...                                                 neighbours=1)
 >>> res = swath.get_sample_from_neighbour_info('nn', area_def.shape, data.ravel(), 
 ...                                            valid_index, index_array)
 
Note the keyword argument **neighbours=1**. This specifies only to consider one neighbour for each 
grid point (the nearest neighbour). Also note **distance_array** is not a required argument for
**get_sample_from_neighbour_info** when using nearest neighbour resampling
    
 