.. _plot:

Plotting with pyresample and Basemap
====================================
Pyresample supports basic integration with Basemap (http://matplotlib.sourceforge.net/basemap).

Displaying data quickly
-----------------------
Pyresample has some convenience functions for displaying data from a single channel. 
The function **plot.show_quicklook** shows a Basemap image of a dataset for a specified AreaDefinition.
The function **plot.save_quicklook** saves the Basemap image directly to file.

**Example usage:**

.. doctest::

 >>> import numpy as np
 >>> from pyresample import load_area, save_quicklook, SwathDefinition
 >>> from pyresample.kd_tree import resample_nearest
 >>> lons = np.zeros(1000)
 >>> lats = np.arange(-80, -90, -0.01)
 >>> tb37v = np.arange(1000)
 >>> area_def = load_area('areas.cfg', 'ease_sh')
 >>> swath_def = SwathDefinition(lons, lats)
 >>> result = resample_nearest(swath_def, tb37v, area_def,
 ...                           radius_of_influence=20000, fill_value=None)
 >>> save_quicklook('tb37v_quick.png', area_def, result, label='Tb 37v (K)')

Assuming **lons**, **lats** and **tb37v** are initialized with real data the result might look something like this:
  .. image:: _static/images/tb37v_quick.png
  
The data passed to the functions is a 2D array matching the AreaDefinition.

The Plate Carree projection
+++++++++++++++++++++++++++
The Plate Carree projection (regular lon/lat grid) is named **eqc** in Proj.4 and **cyl** in Basemap. pyresample uses the Proj.4 name.
Assuming the file **areas.cfg** has the following area definition:

.. code-block:: bash

 REGION: pc_world {
    NAME:    Plate Carree world map
    PCS_ID:  pc_world
    PCS_DEF: proj=eqc
    XSIZE: 640
    YSIZE: 480
    AREA_EXTENT:  (-20037508.34, -10018754.17, 20037508.34, 10018754.17)
 };

**Example usage:**

 >>> import numpy as np 
 >>> from pyresample import load_area, save_quicklook, SwathDefinition
 >>> from pyresample.kd_tree import resample_nearest
 >>> lons = np.zeros(1000)
 >>> lats = np.arange(-80, -90, -0.01)
 >>> tb37v = np.arange(1000)
 >>> area_def = load_area('areas.cfg', 'pc_world')
 >>> swath_def = SwathDefinition(lons, lats)
 >>> result = resample_nearest(swath_def, tb37v, area_def, radius_of_influence=20000, fill_value=None)
 >>> save_quicklook('tb37v_pc.png', area_def, result, num_meridians=0, num_parallels=0, label='Tb 37v (K)')

Assuming **lons**, **lats** and **tb37v** are initialized with real data the result might look something like this:
  .. image:: _static/images/tb37v_pc.png


The Globe projections
+++++++++++++++++++++
From v0.7.12 pyresample can use the geos, ortho and nsper projections with Basemap.
Assuming the file **areas.cfg** has the following area definition for an ortho projection area:

.. code-block:: bash

 REGION: ortho {
   NAME:    Ortho globe
   PCS_ID:  ortho_globe
   PCS_DEF: proj=ortho, a=6370997.0, lon_0=40, lat_0=-40
   XSIZE: 640
   YSIZE: 480
   AREA_EXTENT:  (-10000000, -10000000, 10000000, 10000000) 
 };

**Example usage:**

 >>> import numpy as np 
 >>> from pyresample import load_area, save_quicklook, SwathDefinition
 >>> from pyresample.kd_tree import resample_nearest
 >>> lons = np.zeros(1000)
 >>> lats = np.arange(-80, -90, -0.01)
 >>> tb37v = np.arange(1000)
 >>> area_def = load_area('areas.cfg', 'ortho')
 >>> swath_def = SwathDefinition(lons, lats)
 >>> result = resample_nearest(swath_def, tb37v, area_def, radius_of_influence=20000, fill_value=None)
 >>> save_quicklook('tb37v_ortho.png', area_def, result, num_meridians=0, num_parallels=0, label='Tb 37v (K)')

Assuming **lons**, **lats** and **tb37v** are initialized with real data the result might look something like this:
  .. image:: _static/images/tb37v_ortho.png


Getting a Basemap object
------------------------
In order to make more advanced plots than the preconfigured quicklooks a Basemap object can be generated from an
AreaDefintion using the **plot.area_def2basemap(area_def, **kwargs)** function.

**Example usage:**

.. doctest::

 >>> import numpy as np	
 >>> import matplotlib.pyplot as plt
 >>> from pyresample import load_area, save_quicklook, area_def2basemap, SwathDefinition
 >>> from pyresample.kd_tree import resample_nearest
 >>> lons = np.zeros(1000)
 >>> lats = np.arange(-80, -90, -0.01)
 >>> tb37v = np.arange(1000)
 >>> area_def = load_area('areas.cfg', 'ease_sh')
 >>> swath_def = SwathDefinition(lons, lats)
 >>> result = resample_nearest(swath_def, tb37v, area_def,
 ...                           radius_of_influence=20000, fill_value=None)
 >>> bmap = area_def2basemap(area_def)
 >>> bmng = bmap.bluemarble()
 >>> col = bmap.imshow(result, origin='upper')
 >>> plt.savefig('tb37v_bmng.png', bbox_inches='tight')

Assuming **lons**, **lats** and **tb37v** are initialized with real data the result might look something like this:
  .. image:: _static/images/tb37v_bmng.png
  
Any keyword arguments (not concerning the projection) passed to **plot.area_def2basemap** will be passed
directly to the Basemap initialization.

For more information on how to plot with Basemap please refer to the Basemap and matplotlib documentation.

Limitations
-----------
The pyresample use of Basemap is basically a conversion from a pyresample AreaDefintion to a Basemap object
which allows for correct plotting of a resampled dataset using the **basemap.imshow** function.

Currently only the following set of Proj.4 arguments can be interpreted in the conversion: 
{'proj', 'a', 'b', 'ellps', 'lon_0', 'lat_0', 'lon_1', 'lat_1', 'lon_2', 'lat_2', 'lat_ts'}

Any other Proj.4 parameters will be ignored. 
If the ellipsoid is not defined in terms of 'ellps', 'a' or ('a', 'b') it will default to WGS84.

The xsize and ysize in an AreaDefinition will only be used during resampling when the image data for use in
**basemap.imshow** is created. The actual size and shape of the final plot is handled by matplotlib.
