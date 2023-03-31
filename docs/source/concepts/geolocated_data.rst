Geolocated Data
===============

The data that Pyresample works with typically represent observations on the
Earth (or other spheroid body). Arrays of these data points are most useful
when we know exactly where they are located, how big of an area they represent,
and what their relationship is to the other points near them. Some of these
properties can be very complex, so we may simplify them to make computations
easier and simplify how we work with them.

Points or Pixels
----------------

Pyresample can generally work with arrays of data with any number of
dimensions. Most examples will stick to the basic cases of arrays that
represent a series of individual
points (1D array), an "image" (2D), and in some cases a multi-band image or
volume (3D) array. In all of these cases we still break things down to
individual points and when visualizing them we may represent them as pixels,
but that isn't usually an accurate representation.

Let's imagine a simple space instrument orbiting the Earth
that is able to point at a single location on the Earth and get a temperature
value. We could represent this point by the temperature
and its location as a longitude and a latitude coordinate in degrees, but that
still wouldn't completely define what this temperature really means.

Footprint
^^^^^^^^^

We should also consider the "footprint" or size and shape of that point on
the Earth.
Usually a space-based instrument isn't measuring a single micrometer of the
Earth. More likely it is measuring a region tens, hundreds, or thousands of
meters wide. We could represent this measured region as a disc with a radius,
but another shape (ex. ellipse) could be more accurate depending on things like
the angle that the measurement was taken, the way the instrument makes the
measurement (ex. moving while recording), the way the instrument works, or
the way it works in various space and Earth atmospheric conditions, or many
other complicated situations. All of these potential representations of this
single "point" of data require different numbers of coordinates (single point
versus disc versus ellipse versus bounding box).

In many of the algorithms implemented in Pyresample these points will be
treated as either a single point with no radius of influence (ex. distance
calculations between two points) or they are treated like a square or
rectangular "pixel" of data with a width and height. This ultimately depends
on the algorithm or utility being used and what makes the most sense for that
algorithm.

By treating points this way we are able to quickly work with large arrays of
data. It allows transforming between instrument-centric representations of
data, gridded forecasting models, and rectangular images. Although this isn't
always the most accurate, it serves a large number of use cases.

Elevation
^^^^^^^^^

Another property of our data points that we may want to be concerned about
is the elevation above the Earth. This could be important if our goal
is to combine data from different sources. For example, space-based instruments
may be "seeing" the top of the atmosphere or the tops of clouds, while
ground-based instruments or observations may be using ground-based or lower
altitude measurements. In many basic resampling cases special handling of
elevation is not needed to get valid usable results.

At the time of writing Pyresample does not have any special handling for these
differences. It is up to the user to deal with any differences in elevation if
their use case needs this type of precision. Note that it is common for
satellite-based instruments to have their geolocation coordinates adjusted to
be ground-based. Another type of adjustment related to this is called parallax
correction. At the time of writing Pyresample does not currently have any
parallax correction algorithms implemented, but the
:doc:`Satpy library <satpy:index>` does.

Pixel Spacing
-------------

The two most common structures of geolocated data used with Pyresample are
uniformly spaced grids of pixels (sometimes called "areas") and swaths of
variably spaced pixels. Depending on the structure of our data we are able to
take certain shortcuts in representing it or in how we approach resampling.
For example, if our data are uniformly spaced we don't need a coordinate for
each pixel, we can store the coordinates for one pixel and the offset (size)
to the next pixel. This can save memory (not storing coordinates for every
pixel), but also lets us quickly and efficiently calculate subsets of our
grid (indices, center coordinates, bounding coordinates, etc).

More details on the different types of geolocation structures can be found
in the :doc:`geometries` documentation.

Bounding Polygon
^^^^^^^^^^^^^^^^

Some operations with geolocated data don't always require knowing the location
of every pixel. Doing calculations with every pixel in these cases would require
a lot of memory and execution time and come up with an answer very similar to
if only a handful of bounding coordinates were used instead. For example,
if we wanted to know if two datasets overlapped we can simplify a large set of
coordinates two a few vertices of a bounding polygon.

Note that although working with a polygon is much faster for certain operations,
creating the polygon can still be complex and costly depending on the structure
of the data. For example, in the case of a swath of non-uniformly spaced pixels
we'll need to extract a subset of coordinates from a potentially very large
arrays, possibly loaded from data files on disk (a slow operation).

Model of the Earth
------------------

One of the most important and also most complex aspects of geolocation of data
is the model of the Earth (or other planet) being used. We could easily use
longitude and latitude coordinates, but is that for a spherical Earth or a
ellipsoid? Is the origin (center) of our coordinate system the same as another
dataset's coordinate system? Or are we projecting the ellipsoidal Earth onto a
flat surface and using cartesian (x/y) coordinates?

There are thousands of ways to represent the Earth and we need to know what our
data is using. This is a large complex topic, but the start of understanding it
is discussed in the next concept documentation: :doc:`projections`.
