Geometries
==========

To work with our geolocated data in Pyresample we need a way to define that
geolocation through a combination of the concepts discussed in
:doc:`geolocated_data` and :doc:`projections`. Pyresample defines geolocation
through a set of "geometry" objects that fall into two main categories: areas
and swaths. Areas (a.k.a area definitions) typically represent a grid of
equally-spaced pixels. A swath of pixels on the other hand
represents non-uniformly spaced pixels. See the sections below for more details
on these concepts and how Pyresample understands the information.

Swaths
------

Swaths are a collection of pixels that may or may not be uniformly sized and
spaced. The pixels may be contiguous where the edges of the footprints touch
or be non-contiguous. They could overlap or be spaced far apart. The data in
the array could maintain the topology of the data or could consist of unordered
pixels. The pixels of the swath can each have a different footprint shape
(ex. disc, ellipse, square, etc) and be different sizes. In this way swaths of
pixels can be used to represent any data as a pair of coordinate arrays.
Typically these coordinates are longitude and latitude arrays in degrees. One
real world and common case for using swaths to represent data is for low Earth
orbiting meteorological satellite instruments where due to the scanning pattern
of the instrument the easiest way to specify the locations of the observed values
is individual coordinates. In Pyresample, we represent swaths with the
:class:`~pyresample.geometry.SwathDefinition` class.

At the time of writing, Pyresample does not have a way to define a swath as
having one or more of these properties. It is up to the user to use the tools
that are compatible with their data.
While many utilities and algorithms in Pyresample may accept swaths as inputs,
they may depend on those swaths having specific properties. For example, some
resampling algorithms may assume that a swath's array preserve the geographic
topology of the observed pixels as an optimization. If this isn't true, the
results may not be accurate, and the algorithm will likely not warn you of
this incompatibility. On the other side of this, many algorithms or tools in
Pyresample may simplify the representation of your swath and treat footprints
as individual points in space (ex. for distance calculations) or assume a
less complex footprint (ex. square versus ellipse).

For all its simplicity, defining your data's geolocation as a swath can come
with some unfortunate consequences. In the most basic definition of a swath
with only the longitude and latitude coordinates (and no additional metadata),
in addition to the memory to hold these two arrays, any information we want
about the swath will either require looking at every coordinate or it will
make some assumption about the data. These types of operations can be very
costly compared to using something like an "area" (see below) if that is at
all an option. For example, if we wanted to create a polygon representing the
bounding coordinates of the swath we must assume that the outer edges of our 2D
longitude and latitude arrays actually represent the edge of the swath. A
swath's pixels are not necessarily in a specific order so this may be
an incorrect assumption. Additionally, for large arrays it could take a
long time to compute accurate bounding coordinates.
In some cases, data files may come with longitude and latitude arrays for their
familiarity with the typical users of those data, but are actually representing
gridded data. In these cases it may be more efficient to create an area (see
below sections).

.. warning::

   The use of specific CRSes when working with swaths is a relatively new
   functionality in Pyresample. As such, most operations assume a generic
   WGS84 lon/lat coordinate reference system for any longitude/latitude
   coordinates although it is possible to set a specific CRS in the
   :class:`~pyresample.geometry.SwathDefinition` class.

Areas
-----

An area or area definition represents a grid of contiguous uniformly sized and
spaced pixels. An area is defined on one specific coordinate reference system
(CRS) and therefore the internal units
(ex. pixel size, bounding extents, etc) are in the units of the projection,
such as degrees or meters (see :doc:`projections` for more information).
Due to this strict definition we can represent an area with only a few
properties. For example, in addition to the CRS, we could use:

* extents: Four values representing the outer limit of the bottom, left,
  top, and right pixels of the grid.
* number of pixels: The number of pixels in the X (columns) and Y (rows)
  dimensions.

Although not necessarily accurate it can sometimes be helpful to think of these
points/pixels as squares (see :doc:`geolocated_data`).
Alternatively, we could use measurements like the size of the pixels in the
X and Y dimension or the coordinate of one of the corner pixels. Or instead of
outer extents, we could use the outer pixels' center points.
Pyresample uses the :class:`~pyresample.geometry.AreaDefinition` to contain all
of this information. You can learn about the many different ways to create an
AreaDefinition from the :doc:`../howtos/geometry_utils` guide.

Unlike swaths, an area definition's properties mean we don't have to hold
arrays of data in memory. The order and contiguous nature of the pixels also
means that we can easily get bounding coordinates or create subsets of the
data and area. We also know that pixels don't overlap one another so there is
little concern of artifacts for dividing the area into separate chunks or
segments for parallel processing and then merging the results back together.
Only needing these few parameters to describe a large region means we can also
quickly compare two areas (ex. equality, hashing, etc) or store the definition
in a text format (ex. YAML).

Dynamic Areas
-------------

Dynamic areas are area definitions who are missing one or more of the
properties needed to fully describe the area. For example, if you had an area
definition where you knew the 4 extent values, but not the number of pixels
inside. We can still carry the information we do know (Pyresample uses
:class:`~pyresample.geometry.DynamicAreaDefinition`), but when we actually
want to use it (ex. resampling) we need to provide the missing information in
one way or another. In Pyresample we call this process "freezing" the dynamic
area and we typically determine the information from longitude and latitude
arrays being provided.

A common use case is to have a dynamic area where we know the CRS and the
resolution of each pixel, but we don't know the extents needed to completely
contain our input data when it is resampled. By freezing the dynamic area with
the longitude and latitude arrays of our input we can have output that is
consistent in pixel size and "look" (based on the CRS) between data
cases (ex. orbits of polar-orbiting satellite instrument data).
