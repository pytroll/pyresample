Preprocessing of grids
======================

When resampling is performed repeatedly to the same grid significant execution time can be save by 
preprocessing grid information.

Preprocessing for grid resampling
---------------------------------

Using the function **generate_quick_linesample_arrays** or 
**generate_nearest_neighbour_linesample_arrays** from **pyresample.utils** arrays containing 
the rows and cols indices used to calculate the result in **image.resample_area_quick** or
**resample_area_nearest_neighbour** can be obtained. These can be fed to the method 
**get_array_from_linesample** of an **ImageContainer** object to obtain the resample result.

.. doctest::

 >>> import numpy
 >>> from pyresample import utils, image, geometry
 >>> area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
 ...                                {'a': '6378144.0', 'b': '6356759.0',
 ...                                 'lat_0': '50.00', 'lat_ts': '50.00',
 ...                                 'lon_0': '8.00', 'proj': 'stere'}, 
 ...                                800, 800,
 ...                                [-1370912.72, -909968.64,
 ...                                 1029087.28, 1490031.36])
 >>> msg_area = geometry.AreaDefinition('msg_full', 'Full globe MSG image 0 degrees',
 ...                                'msg_full',
 ...                                {'a': '6378169.0', 'b': '6356584.0',
 ...                                 'h': '35785831.0', 'lon_0': '0',
 ...                                 'proj': 'geos'},
 ...                                3712, 3712,
 ...                                [-5568742.4, -5568742.4,
 ...                                 5568742.4, 5568742.4])
 >>> data = numpy.ones((3712, 3712))
 >>> msg_con = image.ImageContainer(data, msg_area) 
 >>> row_indices, col_indices = \
 ...		utils.generate_nearest_neighbour_linesample_arrays(msg_area, area_def, 50000)
 >>> result = msg_con.get_array_from_linesample(row_indices, col_indices) 

The numpy arrays returned by **generate_*_linesample_arrays** can be and used with the 
**ImageContainer.get_array_from_linesample** method when the same resampling is to be performed 
again thus eliminating the need for calculating the reprojection.

Numpy arrays can be saved and loaded using  **numpy.save** and **numpy.load**.

Preprocessing for swath resampling
----------------------------------

Using the function **generate_cartesian_grid** from **pyresample.utils** an array 
containing the target grid can be obtained.

.. doctest::

 >>> import numpy
 >>> from pyresample import swath, utils, geometry
 >>> area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
 ...                                {'a': '6378144.0', 'b': '6356759.0',
 ...                                 'lat_0': '50.00', 'lat_ts': '50.00',
 ...                                 'lon_0': '8.00', 'proj': 'stere'}, 
 ...                                800, 800,
 ...                                [-1370912.72, -909968.64,
 ...                                 1029087.28, 1490031.36])
 >>> cart_grid = utils.generate_cartesian_grid(area_def)
 >>> lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
 >>> lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
 >>> data = numpy.fromfunction(lambda y, x: y*x, (50, 10))
 >>> result = swath.resample_nearest(lons.ravel(), lats.ravel(), 
 ... data.ravel(), target_area_def=cart_grid, radius_of_influence=50000)

The numpy array **cart_grid** can be saved and be used as **target_area_def** whenever a swath is to be 
resampled to this grid. This will eliminate the need for calculation grid coordinates thus 
decreasing execution time.




