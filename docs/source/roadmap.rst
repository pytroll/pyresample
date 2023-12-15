Roadmap
=======

Roadmap to 2.0!
---------------

Pyresample has had some real growing pains. These have become more and more
apparent as Pytroll developers have worked on Satpy where we had more freedom
to start fresh and create interfaces that worked for us. That development along
with the Pyresample User Survey conducted in 2021 have guided us to a new design
for Pyresample that we hope to release as version 2.0.

Below are the various categories of components of Pyresample and how we see them
existing. In most of the cases for existing interfaces in Pyresample, we expect
things to be backwards compatible or in the extreme cases we want to add new
interfaces alongside the existing ones for an easier transition to the new
functionality. This will mean some deprecated interfaces so that we can make
the user experience more consistent regardless of resampling algorithm or other
use case.

You can track the progress of Pyresample 2.0 by following the issues and pull
requests on the
`v2.0 milestone <https://github.com/pytroll/pyresample/milestone/3>`_ on
GitHub.

What is not planned for 2.0?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Vertical or other higher dimension resampling: This is not a simple problem
  and with the other changes planned for 2.0 we won't be able to look into
  this.
* New resampling algorithms: The 2.0 version bump is not meant for new
  algorithms. We'll keep these more for minor version releases. Version 2.0
  is about larger breaking changes.

Geometry Classes
^^^^^^^^^^^^^^^^

There are currently X types of user-facing geometry objects in Pyresample:

* ~GridDefinition~
* SwathDefinition
* DynamicAreaDefinition
* AreaDefinition
* StackedAreaDefinition

We've realized that the ``GridDefinition`` is actually a special case of an
``AreaDefinition`` and therefore it will be deprecated. The other classes,
as far as purpose, are still needed in some sense and will most likely not
go anywhere. Most changes to these classes will be internal for code
cleanliness.

However, we'd like these classes to be easier to create and use. We'd like to
focus on what is actually required to work with these objects with the rest of
Pyresample. This means separating the "numbers" from the metadata.
AreaDefinitions will no longer require a name or other descriptive information.
You can provide them, but they won't be required.

Going forward we'd like users to focus on using classmethod's to create these
objects. We hope this will provide an easier connection from what information
you have to a useable object. For example, instead of:

.. code-block:: python

    area = AreaDefinition(name, description, proj_id, projection, width, height, area_extent)

You would do this in pyresample 2.0:

.. code-block:: python

    metadata = {"name": name, "description": description}  # optional
    area = AreaDefinition.from_extent_shape(projection, area_extent, (height, width), metadata)

You'll also be allowed to provide arbitrary metadata to swath definitions and
the other geometry types.

Resamplers
^^^^^^^^^^

Currently there are different interfaces for calling the different resampling
options in Pyresample. Sometimes you call a function, sometimes you create a
class and call a "resample" method on it, and sometimes if you want finer control
you call multiple functions and have to pass things between them. In Pyresample
2.0 we want to get things down to a few consistent interfaces all wrapped up
into a series of Resampler classes.

Creating Resamplers
*******************

You'll create resampler classes by doing:

.. code-block:: python

    from pyresample import create_resampler
    resampler = create_resampler(src_geom, dst_geom, resampler='some-resampler-name')

Pyresample 2.0 will maintain a "registry" of available resampler classes that
you can refer to by name or get one by default based on the passed geometries.
This registry of resamplers will also make it easier for users or third-party
libraries to add their own resamplers.

We hope with this basic creation process that we can have more control over
what algorithms support what features and let the user know when something
isn't allowed early on with clear error messages. For example, what
combinations of geometry types are supported by the resampler or what types
of arrays (xarray.DataArray, dask, or numpy) can be provided.

Using Resamplers
****************

Once you have your resampler instance you can resample your data by doing:

.. code-block:: python

    new_data = resampler.resample(data, **kwargs)

That's it. There are of course a lot of options hidden in the ``**kwargs``,
but those will be specific to each algorithm. Our hope is that any
optimizations or conversions that need to happen to get your data resampled
can all be contained in these resampler objects and hopefully require less
from the user.

Alternatively to the ``.resample`` call, users can first call two methods:

.. code-block:: python

    resampler.precompute()
    new_data = resampler.compute(data, **kwargs)

This ``precompute`` method will perform any computations that can be done
without needing the actual "image" data. You can then call ``.compute``
to do the actual resampling. This separation is important when we start talking
about caching (see below).

Caching
*******

One major simplification we're hoping to achieve with Pyresample 2.0 is a
defined set of caching functionality all encapsulated in "Cache" objects.
These objects can be passed to ``create_resampler`` to enable the resampler
to store intermediate computation results for reuse. How and where that storing
is done is up to the specific cache object. It could be in-memory only, or
to zarr datasets on local disk, or to some remote storage.

By calling the ``.precompute`` method, the user will be able to pre-fill this
cache without needing any image data. This will be useful for users using
pyresample in operations where they may want to manually fill the cache before
spawning realtime (time sensitive) processing.

Indexes
^^^^^^^

From our survey we learned that a lot of users use the indexes returned by
``get_neighbour_info`` for their own custom analyses. We recognize this need
and while Cache objects could be written to get the same result, we think
there is a better way. We plan to implement this functionality through a
separate "Index" interface. Like Resamplers, these would provide you a way
of relating a source geometry to a destination geometry. However, these
objects would only be responsible for returning the index information.

We haven't defined the interface for these yet, but hope that having something
separate from resamplers will serve more people.

Xarray and Geoxarray
^^^^^^^^^^^^^^^^^^^^

We would like to support pyresample users who use the xarray and dask libraries
more. Behind the scenes over the last couple years we've added a lot of
dask-based support to pyresample through the Satpy library. We've slowly moved
that functionality over to Pyresample and the Resampler objects mentioned above
are the first defined interface for that. However, there is still a lot of work
to be done to completely take advantage of the parallel nature that dask arrays
provide us.

It should also be easier for users with data in xarray DataArray or Dataset
objects to access pyresample functionality; even without knowing the
metadata that pyresample will need to do some resampling (ex. CRS, extents,
etc). Usually that type of information is held in the metadata of the xarray
object already. New tools are in development to make this information easier
to access; mainly
`the Geoxarray project <https://geoxarray.github.io/latest/>`_.
We will be working on Geoxarray and Pyresample to simplify common resampling
tasks for xarray users.

Documentation
^^^^^^^^^^^^^

The documentation for Pyresample is in need of a lot of love. As Pyresample
has grown the documentation hasn't really been restructured to best present
the new information it has taken on. We hope that as part of Pyresample 2.0
we can clean out the cobwebs and make it easier to find the information you
are looking for.
