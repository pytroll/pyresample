.. testsetup::

   import numpy as np
   lons = np.zeros(1000)
   lats = np.arange(-80, -90, -0.01)
   tb37v = np.arange(1000)
   from pyresample import SwathDefinition
   swath_def = SwathDefinition(lons, lats)
   if plt is not None:
       import matplotlib
       matplotlib.use('agg')

Getting a Cartopy CRS
---------------------

To make more advanced plots than the preconfigured quicklooks Cartopy_ can be
used to work with mapped data alongside matplotlib. The below code is based on
this `Cartopy gallery example`_. Pyresample allows any `AreaDefinition` to be
converted to a Cartopy_ CRS as long as Cartopy_ can represent the
projection. Once an `AreaDefinition` is converted to a CRS object it can be
used like any other Cartopy_ CRS object.

.. doctest::
   :skipif: plt is None or cartopy is None

   >>> import matplotlib.pyplot as plt
   >>> from pyresample.kd_tree import resample_nearest
   >>> from pyresample.geometry import AreaDefinition
   >>> area_id = 'alaska'
   >>> description = 'Alaska Lambert Equal Area grid'
   >>> proj_id = 'alaska'
   >>> projection = {'proj': 'stere', 'lat_0': 62., 'lon_0': -152.5, 'ellps': 'WGS84', 'units': 'm'}
   >>> width = 2019
   >>> height = 1463
   >>> area_extent = (-757214.993104, -485904.321517, 757214.993104, 611533.818622)
   >>> area_def = AreaDefinition(area_id, description, proj_id, projection,
   ...                           width, height, area_extent)
   >>> result = resample_nearest(swath_def, tb37v, area_def, radius_of_influence=20000, fill_value=None)
   >>> crs = area_def.to_cartopy_crs()
   >>> fig, ax = plt.subplots(subplot_kw=dict(projection=crs))
   >>> coastlines = ax.coastlines()  # doctest: +SKIP
   >>> ax.set_global()
   >>> img = plt.imshow(result, transform=crs, extent=crs.bounds, origin='upper')
   >>> cbar = plt.colorbar()
   >>> fig.savefig('amsr2_tb37v_cartopy.png')


Assuming **lons**, **lats**, and **i04_data** are initialized with real data
the result might look something like this:

  .. image:: /_static/images/amsr2_tb37v_cartopy.png

Getting a Basemap object
------------------------

.. warning::

    Basemap is no longer maintained. Cartopy_ (see above) should be used
    instead. Basemap does not support Matplotlib 3.0+ either.

In order to make more advanced plots than the preconfigured quicklooks a Basemap object can be generated from an
AreaDefinition using the :func:`area_def2basemap <pyresample.plot.area_def2basemap>` function.

**Example usage:**

.. doctest::
   :skipif: plt is None or Basemap is None

   >>> import matplotlib.pyplot as plt
   >>> from pyresample.kd_tree import resample_nearest
   >>> from pyresample.geometry import AreaDefinition
   >>> area_id = 'ease_sh'
   >>> description = 'Antarctic EASE grid'
   >>> proj_id = 'ease_sh'
   >>> projection = {'proj': 'laea', 'lat_0': -90, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'}
   >>> width = 425
   >>> height = 425
   >>> area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
   >>> area_def = AreaDefinition(area_id, description, proj_id, projection,
   ...                           width, height, area_extent)
   >>> from pyresample import area_def2basemap
   >>> result = resample_nearest(swath_def, tb37v, area_def,
   ...                           radius_of_influence=20000, fill_value=None)
   >>> bmap = area_def2basemap(area_def) # doctest: +SKIP
   >>> bmng = bmap.bluemarble() # doctest: +SKIP
   >>> col = bmap.imshow(result, origin='upper', cmap='RdBu_r') # doctest: +SKIP
   >>> plt.savefig('tb37v_bmng.png', bbox_inches='tight') # doctest: +SKIP


Assuming **lons**, **lats** and **tb37v** are initialized with real data as in
the previous examples the result might look something like this:

  .. image:: /_static/images/tb37v_bmng.png

Any keyword arguments (not concerning the projection) passed to
**plot.area_def2basemap** will be passed directly to the Basemap
initialization.

For more information on how to plot with Basemap please refer to the Basemap
and matplotlib documentation.


Adding background maps with Cartopy
-----------------------------------

As mentioned in the above warning Cartopy_ should be used rather than Basemap as
the latter is not maintained anymore.

The above image can be generated using Cartopy_ instead by utilizing the method
`to_cartopy_crs` of the `AreaDefinition` object.

**Example usage:**

.. doctest::
   :skipif: plt is None or cartopy is None

   >>> from pyresample.kd_tree import resample_nearest
   >>> import matplotlib.pyplot as plt
   >>> result = resample_nearest(swath_def, tb37v, area_def,
   ...                           radius_of_influence=20000, fill_value=None)
   >>> crs = area_def.to_cartopy_crs()
   >>> ax = plt.axes(projection=crs)
   >>> ax.background_img(name='BM')  # doctest: +SKIP
   >>> plt.imshow(result, transform=crs, extent=crs.bounds, origin='upper', cmap='RdBu_r')  # doctest: +SKIP
   >>> plt.savefig('tb37v_bmng.png', bbox_inches='tight')  # doctest: +SKIP


The above provides you have the Bluemarble background data available in the
Cartopy_ standard place or in a directory pointed to by the environment
parameter `CARTOPY_USER_BACKGROUNDS`.

With real data (same AMSR-2 as above) this might look like this:

  .. image:: /_static/images/tb37v_bmng_cartopy.png

.. _Satpy: http://www.github.com/pytroll/satpy
.. _`Cartopy gallery example`: http://scitools.org.uk/cartopy/docs/v0.16/gallery/geostationary.html
.. _Cartopy: http://scitools.org.uk/cartopy/
