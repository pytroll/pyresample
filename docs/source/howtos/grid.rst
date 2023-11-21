Resampling of gridded data
==========================

Pyresample can be used to resample from an existing grid to another. Nearest neighbour resampling is used.

pyresample.image
----------------

.. note::
   The pyresample.image module is deprecated.  Please use `pyresample.kd_tree` or `pyresample.bilinear`
   instead.

A grid can be stored in an object of type **ImageContainer** along with its area definition.
An object of type **ImageContainer** allows for calculating resampling using preprocessed arrays
using the method **get_array_from_linesample**

Resampling can be done using descendants of **ImageContainer** and calling their **resample** method.

An **ImageContainerQuick** object allows for the grid to be resampled to a new area defintion
using an approximate (but fast) nearest neighbour method.
Resampling an object of type **ImageContainerQuick** returns a new object of type **ImageContainerQuick**.

An **ImageContainerNearest** object allows for the grid to be resampled to a new area defintion (or swath definition)
using an accurate kd-tree method.
Resampling an object of type **ImageContainerNearest** returns a new object of
type **ImageContainerNearest**.

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
 >>> msg_area = geometry.AreaDefinition('msg_full', 'Full globe MSG image 0 degrees',
 ...                                'msg_full',
 ...                                {'a': '6378169.0', 'b': '6356584.0',
 ...                                 'h': '35785831.0', 'lon_0': '0',
 ...                                 'proj': 'geos'},
 ...                                3712, 3712,
 ...                                [-5568742.4, -5568742.4,
 ...                                 5568742.4, 5568742.4])
 >>> data = np.ones((3712, 3712))
 >>> msg_con_quick = image.ImageContainerQuick(data, msg_area)
 >>> area_con_quick = msg_con_quick.resample(area_def)
 >>> result_data_quick = area_con_quick.image_data
 >>> msg_con_nn = image.ImageContainerNearest(data, msg_area, radius_of_influence=50000)
 >>> area_con_nn = msg_con_nn.resample(area_def)
 >>> result_data_nn = area_con_nn.image_data

Data is assumed to be a numpy array of shape (rows, cols) or (rows, cols, channels).

Masked arrays can be used as data input. In order to have undefined pixels masked out instead of
assigned a fill value set **fill_value=None** when calling **resample_area_***.

Using **ImageContainerQuick** the risk of image artifacts increases as the distance
from source projection center increases.

The constructor argument **radius_of_influence** to **ImageContainerNearest** specifices the maximum
distance to search for a neighbour for each point in the target grid. The unit is meters.

The constructor arguments of an ImageContainer object can be changed as attributes later

.. doctest::

 >>> import numpy as np
 >>> from pyresample import image, geometry
 >>> msg_area = geometry.AreaDefinition('msg_full', 'Full globe MSG image 0 degrees',
 ...                                'msg_full',
 ...                                {'a': '6378169.0', 'b': '6356584.0',
 ...                                 'h': '35785831.0', 'lon_0': '0',
 ...                                 'proj': 'geos'},
 ...                                3712, 3712,
 ...                                [-5568742.4, -5568742.4,
 ...                                 5568742.4, 5568742.4])
 >>> data = np.ones((3712, 3712))
 >>> msg_con_nn = image.ImageContainerNearest(data, msg_area, radius_of_influence=50000)
 >>> msg_con_nn.radius_of_influence = 45000
 >>> msg_con_nn.fill_value = -99

Multi channel images
********************

If the dataset has several channels the last index of the data array specifies the channels

.. doctest::

 >>> import numpy as np
 >>> from pyresample import image, geometry
 >>> msg_area = geometry.AreaDefinition('msg_full', 'Full globe MSG image 0 degrees',
 ...                                'msg_full',
 ...                                {'a': '6378169.0', 'b': '6356584.0',
 ...                                 'h': '35785831.0', 'lon_0': '0',
 ...                                 'proj': 'geos'},
 ...                                3712, 3712,
 ...                                [-5568742.4, -5568742.4,
 ...                                 5568742.4, 5568742.4])
 >>> channel1 = np.ones((3712, 3712))
 >>> channel2 = np.ones((3712, 3712)) * 2
 >>> channel3 = np.ones((3712, 3712)) * 3
 >>> data = np.dstack((channel1, channel2, channel3))
 >>> msg_con_nn = image.ImageContainerNearest(data, msg_area, radius_of_influence=50000)


Segmented resampling
********************

Pyresample calculates the result in segments in order to reduce memory footprint. This is controlled by the **segments** contructor keyword argument. If no **segments** argument is given pyresample will estimate the number of segments to use.

Forcing quick resampling to use 4 resampling segments:

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
 >>> msg_area = geometry.AreaDefinition('msg_full', 'Full globe MSG image 0 degrees',
 ...                                'msg_full',
 ...                                {'a': '6378169.0', 'b': '6356584.0',
 ...                                 'h': '35785831.0', 'lon_0': '0',
 ...                                 'proj': 'geos'},
 ...                                3712, 3712,
 ...                                [-5568742.4, -5568742.4,
 ...                                 5568742.4, 5568742.4])
 >>> data = np.ones((3712, 3712))
 >>> msg_con_quick = image.ImageContainerQuick(data, msg_area, segments=4)
 >>> area_con_quick = msg_con_quick.resample(area_def)

Constructor arguments
*********************
The full list of constructor arguments:

 **ImageContainerQuick**:

* image_data : Dataset. Masked arrays can be used.
* geo_def : Geometry definition.
* fill_value (optional) : Fill value for undefined pixels. Defaults to 0. If set to **None** they will be masked out.
* nprocs (optional) : Number of processor cores to use. Defaults to 1.
* segments (optional) : Number of segments to split resampling in. Defaults to auto estimation.

 **ImageContainerNearest**:

* image_data : Dataset. Masked arrays can be used.
* geo_def : Geometry definition.
* radius_of_influence : Cut off radius in meters when considering neighbour pixels.
* epsilon (optional) : The distance to a found value is guaranteed to be no further than (1 + eps) times the distance to the correct neighbour.
* fill_value (optional) : Fill value for undefined pixels. Defaults to 0. If set to **None** they will be masked out.
* reduce_data (optional) : Apply geographic reduction of dataset before resampling. Defaults to True
* nprocs (optional) : Number of processor cores to use. Defaults to 1.
* segments (optional) : Number of segments to split resampling in. Defaults to auto estimation.

Preprocessing of grid resampling
*********************************
For preprocessing of grid resampling see :ref:`preproc`
