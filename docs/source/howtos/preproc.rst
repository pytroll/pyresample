.. _preproc:

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
