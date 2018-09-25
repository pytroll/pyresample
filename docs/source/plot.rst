.. _plot:

Plotting with pyresample and Cartopy
====================================

Pyresample supports basic integration with Cartopy
(http://scitools.org.uk/cartopy/).

Displaying data quickly
-----------------------
Pyresample has some convenience functions for displaying data from a single
channel. The function **plot.show_quicklook** shows a Cartopy generated image
of a dataset for a specified AreaDefinition. The function
**plot.save_quicklook** saves the Cartopy image directly to file.

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

From v0.7.12 pyresample can use the geos, ortho and nsper projections with
Basemap. Starting with v1.9.0 quicklooks are now generated with Cartopy which
should also work with these projections. Assuming the file **areas.cfg** has
the following area definition for an ortho projection area:

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


Getting a Cartopy CRS
---------------------

To make more advanced plots than the preconfigured quicklooks Cartopy can be
used to work with mapped data alongside matplotlib. The below code is based
on
`this <http://scitools.org.uk/cartopy/docs/v0.16/gallery/geostationary.html>`_
Cartopy example. Pyresample allows any `AreaDefinition` to be converted to a
Cartopy CRS as long as Cartopy can represent the projection. Once an
AreaDefinition is converted to a CRS object it can be used like any other
Cartopy CRS object.

 >>> import numpy as np
 >>> import matplotlib.pyplot as plt
 >>> from pyresample import load_area, save_quicklook, SwathDefinition
 >>> from pyresample.kd_tree import resample_nearest
 >>> lons = np.zeros(1000)
 >>> lats = np.arange(-80, -90, -0.01)
 >>> i04_data = np.arange(1000)
 >>> swath_def = SwathDefinition(lons, lats)
 >>> area_def = swath_def.compute_optimal_bb_area({'proj': 'lcc', 'lon_0': -95., 'lat_0': 25., 'lat_1': 25., 'lat_2': 25.})
 >>> result = resample_nearest(swath_def, i04_data, area_def,
 ...                           radius_of_influence=20000, fill_value=None)
 >>> crs = area_def.to_cartopy_crs()
 >>> ax = plt.axes(projection=crs)
 >>> ax.coastlines()
 >>> ax.set_global()
 >>> plt.imshow(data, transform=crs, extent=crs.bounds, origin='upper')
 >>> plt.colorbar()
 >>> plt.savefig('viirs_i04_cartopy.png')

Assuming **lons**, **lats**, and **i04_data** are initialized with real data
the result might look something like this:

  .. image:: _static/images/viirs_i04_cartopy.png

Getting a Basemap object
------------------------

.. warning::

    Basemap is no longer maintained. Cartopy (see above) should be used
    instead. Basemap does not support Matplotlib 3.0+ either.

In order to make more advanced plots than the preconfigured quicklooks a Basemap object can be generated from an
AreaDefinition using the **plot.area_def2basemap(area_def, **kwargs)** function.

**Example usage:**

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
