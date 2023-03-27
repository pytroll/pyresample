Projections
===========

One of the more complex topics when working with geolocated data is geographic
projections and the conversion or resampling of data between projections.
You may also see the term projection referred to as a Projected Coordinate
Reference System (CRS) or just CRS. The below documentation can be seen as
a primer, but it is still quite limited for this complex topic.

The book "Map Projections" by Battersby [#]_ describes map projections:

.. code-block:: text

    Map projection is the process of transforming angular
    (spherical / elliptical) coordinates into planar coordinates. All map
    projections introduce distortion (e.g., to areas, angles, distances) in the
    resulting planar coordinates. Understanding what, where, and how much
    distortion is introduced is an important consideration for spatial
    computations and visual interpretation of spatial patterns, as well as for
    general aesthetics of any map.

Spatial projections describe how to create a flat 2D version of our round 3D
Earth that may be easier to work with or describe than the original
representation used by our data.

.. image:: http://gistbok.ucgis.org/sites/default/files/figure2-projections.png
   :width: 450px
   :target: http://gistbok.ucgis.org/bok-topics/map-projections

These projections or coordinate reference systems will consist of things like
a model of the Earth (datum/ellipsoid), a reference or center longitude, a
reference or center latitude, and the type of shape or algorithm used to
transform points to and from the projection. For example, a geostationary
projection might "see" the Earth like:

.. image:: https://proj.org/_images/geos.png
   :width: 300px
   :target: https://proj.org/operations/projections/geos.html

Or a lambert conformal conic (LCC) projection might "see" it like:

.. image:: https://proj.org/_images/lcc.png
   :width: 300px
   :target: https://proj.org/operations/projections/lcc.html

Changing the parameters for a particular CRS will change what region of the
Earth is covered and what level of distortion is seen compared to the real
Earth.

Properly defining the projection of your data is important in order to properly
compare data from multiple sources. Two pixels at longitude 0 and latitude 0
aren't actually representing the same location if those coordinates are defined
for different CRSes. Typically to make comparison the easiest it can be
coordinates are transformed or data resampled to one single CRS.
For example, data from a forecast model may be defined
on a longitude/latitude projection with coordinates specified in degrees. Data
from a geostationary satellite might be on a geostationary projection with
coordinates in meters. If you wanted to combine these data you'd need to
transform the coordinates from one projection to the other.

.. [#]

   Battersby, S. (2017). Map Projections. The Geographic Information Science &
   Technology Body of Knowledge (2nd Quarter 2017 Edition), John P. Wilson (ed.). DOI: 10.22224/gistbok/2017.2.7

Projection definitions
----------------------

There are many different types of projections available in Pyresample.
Pyresample uses the :doc:`pyproj <pyproj:index>` library's ``CRS`` object to define
all of its Coordinate Reference Systems and the transformation of coordinates
between them. Pyproj depends on the lower-level :doc:`PROJ <proj:index>` C++
library which defines the actual coordinate transformation algorithms and
maintains definitions for predefined CRSes (ex. EPSG codes). You can find a
list of the supported PROJ definitions :doc:`here <proj:operations/projections/index>`.

Anywhere in Pyresample that a CRS is defined a :class:`~pyproj.crs.CRS` object
from ``pyproj`` should be supported. This means that many different forms of
defining CRS objects are available. See the pyproj
:doc:`Getting Started <pyproj:examples>` documentation for some examples of
the options.

Lastly, a projection is not the only thing that is needed to describe where
your data is on the Earth. See the next section on :doc:`geometries` for
information on how pixel size and geographic extents can be combined with a CRS
to define these locations.
