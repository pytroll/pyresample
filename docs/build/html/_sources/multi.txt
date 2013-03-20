.. _multi:

Using multiple processor cores and pykdtree
===========================================

Multi core processing
*********************

Bottlenecks of pyresample can be executed in parallel. Parallel computing can be executed if the 
pyresample function has the **nprocs** keyword argument. **nprocs** specifies the number of processes 
to be used for calculation. If a class takes the constructor argument **nprocs** this sets **nprocs** for
all methods of this class

Example of resampling in parallel using 4 processes:

.. doctest::

 >>> import numpy
 >>> from pyresample import kd_tree, geometry
 >>> area_def = geometry.AreaDefinition('areaD', 'Europe (3km, HRV, VTC)', 'areaD',
 ...                                {'a': '6378144.0', 'b': '6356759.0',
 ...                                 'lat_0': '50.00', 'lat_ts': '50.00',
 ...                                 'lon_0': '8.00', 'proj': 'stere'}, 
 ...                                800, 800,
 ...                                [-1370912.72, -909968.64,
 ...                                 1029087.28, 1490031.36])
 >>> data = numpy.fromfunction(lambda y, x: y*x, (50, 10))
 >>> lons = numpy.fromfunction(lambda y, x: 3 + x, (50, 10))
 >>> lats = numpy.fromfunction(lambda y, x: 75 - y, (50, 10))
 >>> swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
 >>> result = kd_tree.resample_nearest(swath_def, data.ravel(),
 ... area_def, radius_of_influence=50000, nprocs=4)

Note: Do not use more processes than available processor cores. As there is a process creation overhead 
there might be neglible performance improvement using say 8 compared to 4 processor cores. 
Test on the actual system to determine the most sensible number of processes to use. 

Using pykdtree
**************

As of pyresample v0.8.0 pykdtree can be used as backend instead of scipy. 
This enables significant speedups for large datasets.

pykdtree is used as a drop-in replacement for scipy. If it's available it will be used otherwise scipy will be used.
To check which backend is active for your pyresample installation do:

 >>> import pyresample as pr
 >>> pr.kd_tree.which_kdtree()

which returns either 'pykdtree' or 'scipy.spatial'.

Please refere to pykdtree_ for installation description.

If pykdtree is built with OpenMP support the number of threads is controlled with the standard OpenMP environment variable OMP_NUM_THREADS.
The *nprocs* argument has no effect on pykdtree.

.. _pykdtree: https://github.com/storpipfugl/pykdtree