
.. testsetup::

   from pyresample.geometry import AreaDefinition
   area_id = 'ease_sh'
   description = 'Antarctic EASE grid'
   proj_id = 'ease_sh'
   projection = {'proj': 'laea', 'lat_0': -90, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'}
   width = 425
   height = 425
   area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
   area_def = AreaDefinition(area_id, description, proj_id, projection,
                             width, height, area_extent)
   import numpy as np
   lons = np.zeros(1000)
   lats = np.arange(-80, -90, -0.01)
   tb37v = np.arange(1000)
   from pyresample import SwathDefinition
   swath_def = SwathDefinition(lons, lats)
   if plt is not None:
       import matplotlib
       matplotlib.use('agg')

The Plate Carree projection
+++++++++++++++++++++++++++
The Plate Carree projection (regular lon/lat grid) is named **eqc** in
Proj.4. Pyresample uses the Proj.4 naming.

Assuming the file **areas.yaml** has the following area definition:

.. code-block:: yaml

  pc_world:
    description: Plate Carree world map
    projection:
      proj: eqc
      ellps: WGS84
    shape:
      height: 480
      width: 640
    area_extent:
      lower_left_xy: [-20037508.34, -10018754.17]
      upper_right_xy: [20037508.34, 10018754.17]


**Example usage:**

.. testsetup::

   from pyresample.area_config import load_area_from_string
   area_def = load_area_from_string("""
   pc_world:
     description: Plate Carree world map
     projection:
       proj: eqc
       ellps: WGS84
     shape:
       height: 480
       width: 640
     area_extent:
       lower_left_xy: [-20037508.34, -10018754.17]
       upper_right_xy: [20037508.34, 10018754.17]
    """, "pc_world")

.. doctest::
   :skipif: plt is None or cartopy is None

   >>> import matplotlib.pyplot as plt
   >>> from pyresample import load_area, save_quicklook
   >>> from pyresample.kd_tree import resample_nearest
   >>> area_def = load_area('areas.yaml', 'pc_world')  # doctest: +SKIP
   >>> result = resample_nearest(swath_def, tb37v, area_def, radius_of_influence=20000, fill_value=None)
   >>> save_quicklook('tb37v_pc.png', area_def, result, num_meridians=None, num_parallels=None, label='Tb 37v (K)')  # doctest: +SKIP

Assuming **lons**, **lats** and **tb37v** are initialized with real data (like
above we use AMSR-2 data in this example) the result might look something like
this:

  .. image:: /_static/images/tb37v_pc.png


The Globe projections
+++++++++++++++++++++

From v0.7.12 pyresample can use the geos, ortho and nsper projections with
Basemap. Starting with v1.9.0 quicklooks are now generated with Cartopy_ which
should also work with these projections. Again assuming the area-config file
**areas.yaml** has the following definition for an ortho projection area:

.. code-block:: bash

  ortho:
    description: Ortho globe
    projection:
      proj: ortho
      lon_0: 40.
      lat_0: -40.
      a: 6370997.0
    shape:
      height: 480
      width: 640
    area_extent:
      lower_left_xy: [-10000000, -10000000]
      upper_right_xy: [10000000, 10000000]

**Example usage:**

.. testsetup::

   from pyresample.area_config import load_area_from_string
   area_def = load_area_from_string("""
   ortho:
     description: Ortho globe
     projection:
       proj: ortho
       lon_0: 40.
       lat_0: -40.
       a: 6370997.0
     shape:
       height: 480
       width: 640
     area_extent:
       lower_left_xy: [-10000000, -10000000]
       upper_right_xy: [10000000, 10000000]
   """, "ortho")

.. doctest::
   :skipif: plt is None or cartopy is None

   >>> from pyresample import load_area, save_quicklook, SwathDefinition
   >>> from pyresample.kd_tree import resample_nearest
   >>> from pyresample import load_area
   >>> area_def = load_area('areas.yaml', 'ortho') # doctest: +SKIP
   >>> swath_def = SwathDefinition(lons, lats)
   >>> result = resample_nearest(swath_def, tb37v, area_def, radius_of_influence=20000, fill_value=None)
   >>> save_quicklook('tb37v_ortho.png', area_def, result, num_meridians=None, num_parallels=None, label='Tb 37v (K)')  # doctest: +SKIP

Assuming **lons**, **lats** and **tb37v** are initialized with real data, like
in the above examples, the result might look something like this:

  .. image:: /_static/images/tb37v_ortho.png

.. _Cartopy: http://scitools.org.uk/cartopy/
