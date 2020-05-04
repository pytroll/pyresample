.. _plot_quick_display:

Plotting with pyresample and Cartopy
====================================

Pyresample supports basic integration with Cartopy_.

Displaying data quickly
-----------------------
Pyresample has some convenience functions for displaying data from a single
channel. The function **plot.show_quicklook** shows a Cartopy generated image
of a dataset for a specified AreaDefinition. The function
**plot.save_quicklook** saves the Cartopy image directly to file.

**Example usage:**

In this simple example below we use GCOM-W1 AMSR-2 data loaded using Satpy_. Of
course Satpy_ facilitates the handling of these data in an even easier way, but
the below example can be useful if you have some data that are yet not
supported by Satpy_. All you need are a set of geo-referenced values
(longitudes and latitudes and corresponding geophysical values).

First we read in the data with Satpy_:

 >>> from satpy.scene import Scene # doctest: +SKIP
 >>> from glob import glob # doctest: +SKIP
 >>> SCENE_FILES = glob("./GW1AM2_20191122????_156*h5") # doctest: +SKIP
 >>> scn = Scene(reader='amsr2_l1b', filenames=SCENE_FILES) # doctest: +SKIP
 >>> scn.load(["btemp_36.5v"]) # doctest: +SKIP
 >>> lons, lats = scn["btemp_36.5v"].area.get_lonlats() # doctest: +SKIP
 >>> tb37v = scn["btemp_36.5v"].data.compute() # doctest: +SKIP

Data for this example can be downloaded from zenodo_.
If you have your own data, or just want to see that the example code here runs, you can
set the three arrays :code:`lons`, :code:`lats` and :code:`tb37v` accordingly, e.g.:

.. doctest::
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

   >>> import numpy as np
   >>> lons = np.zeros(1000)
   >>> lats = np.arange(-80, -90, -0.01)
   >>> tb37v = np.arange(1000)

But here we go on with the loaded AMSR-2 data. Make sure you have an :code:`areas.yaml`
file that defines the :code:`ease_sh` area, or see
:ref:`the area definition section<area-definitions>` on how to define one.

 >>> from pyresample import load_area, save_quicklook, SwathDefinition
 >>> from pyresample.kd_tree import resample_nearest
 >>> area_def = load_area('areas.yaml', 'ease_sh') # doctest: +SKIP
 >>> swath_def = SwathDefinition(lons, lats)
 >>> result = resample_nearest(swath_def, tb37v, area_def,
 ...                           radius_of_influence=20000, fill_value=None)
 >>> save_quicklook('tb37v_quick.png', area_def, result, label='Tb 37v (K)')
 
Assuming **lons**, **lats** and **tb37v** are initialized with real data (as in
the above Satpy_ example) the result might look something like this:

  .. image:: _static/images/tb37v_quick.png
  
The data passed to the functions is a 2D array matching the AreaDefinition.

             
.. _Satpy: http://www.github.com/pytroll/satpy
.. _zenodo: https://doi.org/10.5281/zenodo.3553696
.. _`Cartopy gallery example`: http://scitools.org.uk/cartopy/docs/v0.16/gallery/geostationary.html
.. _Cartopy: http://scitools.org.uk/cartopy/
