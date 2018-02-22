[![Build Status](https://travis-ci.org/pytroll/pyresample.svg?branch=master)](https://travis-ci.org/pytroll/pyresample)
[![Build status](https://ci.appveyor.com/api/projects/status/10qdrecp45rgkf73/branch/master?svg=true)](https://ci.appveyor.com/project/davidh-ssec/pyresample-ly2q0/branch/master)


Python package for geospatial resampling
----------------------------------------

Resampling (reprojection) of geospatial image data in Python.
Pyresample uses a kd-tree approach for resampling. 
Pyresample is designed for resampling of remote sensing data and supports resampling from both fixed grids and geolocated swath data. 
Several types of resampling are supported including nearest neighbour, gaussian weighting and weighting with a user defined radial function.
Pyresample works with Numpy arrays including support for masked arrays.
Support for parallel resampling using multiple processor cores.
Plotting capablity using Basemap. As of v0.8.0 [pykdtree](https://github.com/storpipfugl/pykdtree) can be used to speed up processing.

Pyresample is tested with Python 2.7, 3.4, 3.5, and 3.6.

Note: For numpy >= 1.6.2 use pyresample >= 0.7.13  

[Documentation](https://pyresample.readthedocs.org/en/latest/)
Look at [pytroll.org](http://pytroll.org/) for more information.


===News===
  * *2015-02-03*: Pyresample-1.1.3 released. Switch to LGPLv3.

  * *2014-12-17*: Pyresample-1.1.2 released. Fix to allow tests to run on travis.

  * *2014-12-10*: Pyresample-1.1.1 released. Wrapping of longitudes and latitudes is now implemented.

  * *2013-10-23*: Pyresample-1.1.0 released. Added option for calculating uncertainties for weighted kd-tree resampling. From now on pyresample will adhere to [http://semver.org/ semantic versioning].

  * *2013-07-03*: Pyresample-1.0.0 released. Minor API change to the geometry.py module as the boundary variable is removed and replaced by proj_x_coords and proj_y_coords. Caching scheme removed from projection coordinate calculation in geometry.py as it introduced excessive complications. The numexpr package is now used for minor bottleneck optimization if available. Version number bumped to 1.0.0 as pyresample has been running stable in production environments for several years now.
   
  * *2013-03-20*: Pyresample-0.8.0 released. Enables use of pykdtree. Fixes projection handling for 'latlong' projection.

  * *2013-01-21*: Pyresample-0.7.13 released. Fixes numpy incompability introduced with numpy v1.6.2

  * *2012-10-18*: Pyresample-0.7.12 released. Better integration with Basemap with support for plotting using globe projections (geos, ortho and nspere). Documentation updated with correct description of the epsilon parameter.

  * *2012-07-03*: Pyresample-0.7.11 released. Support for plotting in Plate Carree projection and bugfixes for meridians and parallels in plots. Added utils.fwhm2sigma convenience function for use in Gauss resampling.   
