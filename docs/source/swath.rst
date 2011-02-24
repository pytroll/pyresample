.. _swath:

Resampling of swath data
========================

Pyresample can be used to resample a swath dataset to a grid, a grid to a swath or a swath to another swath. 
Resampling can be done using nearest neighbour method, Guassian weighting, weighting with an arbitrary radial function.

pyresample.image
----------------
The ImageContainerNearest class can be used for nearest neighbour resampling of swaths as well as grids.

.. doctest::

 >>> import numpy as np
 >>> from pyresample import image, geometry
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
 >>> swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
 >>> swath_con = image.ImageContainerNearest(data, swath_def, radius_of_influence=5000)
 >>> area_con = swath_con.resample(area_def)
 >>> result = area_con.image_data

For other resampling types or splitting the process in two steps use the functions in **pyresample.swath** described below. 

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
 >>> from pyresample import kd_tree, geometry
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
 >>> swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
 >>> result = kd_tree.resample_nearest(swath_def, data,
 ... area_def, radius_of_influence=50000, epsilon=100)

If the arguments **swath_def** and **area_def** where switched (and **data** matched the dimensions of **area_def**) the grid of **area_def**
would be resampled to the swath defined by **swath_def**.  

Note the keyword arguments:

* **radius_of_influence**: The radius around each grid pixel in meters to search for neighbours in the swath.
* **epsilon**: Allowed uncertainty in nearest neighbour search in meters. Allowing for uncertanty decreases execution time.

If **data** is a masked array the mask will follow the neighbour pixel assignment.

If there are multiple channels in the dataset the **data** argument should be of the shape of the lons and lat arrays 
with the channels along the last axis e.g. (rows, cols, channels). Note: the convention of pyresample < 0.7.4 is to pass
**data** in the form of (number_of_data_points, channels) is still accepted.

.. doctest::

 >>> import numpy as np
 >>> from pyresample import kd_tree, geometry
 >>> area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
 ...                                {'a': '6378144.0', 'b': '6356759.0',
 ...                                 'lat_0': '50.00', 'lat_ts': '50.00',
 ...                                 'lon_0': '8.00', 'proj': 'stere'}, 
 ...                                800, 800,
 ...                                [-1370912.72, -909968.64,
 ...                                 1029087.28, 1490031.36])
 >>> channel1 = np.fromfunction(lambda y, x: y*x, (50, 10))
 >>> channel2 = np.fromfunction(lambda y, x: y*x, (50, 10)) * 2
 >>> channel3 = np.fromfunction(lambda y, x: y*x, (50, 10)) * 3
 >>> data = np.dstack((channel1, channel2, channel3))
 >>> lons = np.fromfunction(lambda y, x: 3 + x, (50, 10))
 >>> lats = np.fromfunction(lambda y, x: 75 - y, (50, 10))
 >>> swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
 >>> result = kd_tree.resample_nearest(swath_def, data,
 ... area_def, radius_of_influence=50000, epsilon=100) 

For nearest neighbour resampling the class **image.ImageContainerNearest** can be used as well as **kd_tree.resample_nearest**

resample_gauss
**************

Function for resampling using nearest Gussian weighting. The Gauss weigh function is defined as exp(-dist^2/sigma^2).

Example showing how to resample a generated swath dataset to a grid using Gaussian weighting:

.. doctest::

 >>> import numpy as np
 >>> from pyresample import kd_tree, geometry
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
 >>> swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
 >>> result = kd_tree.resample_gauss(swath_def, data, 
 ... area_def, radius_of_influence=50000, sigmas=25000)

If more channels are present in **data** the keyword argument **sigmas** must be a list containing a sigma for each channel.

If **data** is a masked array any pixel in the result data that has been "contaminated" by weighting of a masked pixel is masked.

resample_custom
***************

Function for resampling using arbitrary radial weight functions.

Example showing how to resample a generated swath dataset to a grid using an arbitrary radial weight function:

.. doctest::

 >>> import numpy as np
 >>> from pyresample import kd_tree, geometry 
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
 >>> swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
 >>> wf = lambda r: 1 - r/100000.0
 >>> result  = kd_tree.resample_custom(swath_def, data,
 ...  area_def, radius_of_influence=50000, weight_funcs=wf)

If more channels are present in **data** the keyword argument **weight_funcs** must be a list containing a radial function for each channel.

If **data** is a masked array any pixel in the result data that has been "contaminated" by weighting of a masked pixel is masked.

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
 >>> from pyresample import kd_tree, geometry
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
 >>> swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
 >>> valid_input_index, valid_output_index, index_array, distance_array = \
 ...                        kd_tree.get_neighbour_info(swath_def, 
 ...                               	                   area_def, 50000,  
 ...                                                   neighbours=1)
 >>> res = kd_tree.get_sample_from_neighbour_info('nn', area_def.shape, data, 
 ...                                              valid_input_index, valid_output_index,
 ...                                              index_array)
 
Note the keyword argument **neighbours=1**. This specifies only to consider one neighbour for each 
grid point (the nearest neighbour). Also note **distance_array** is not a required argument for
**get_sample_from_neighbour_info** when using nearest neighbour resampling

Segmented resampling
********************
Whenever a resampling function takes the keyword argument **segments** the number of segments to split the resampling process in can be specified. This affects the memory footprint of pyresample. If the value of **segments** is left to default pyresample will estimate the number of segments to use. 
    
 