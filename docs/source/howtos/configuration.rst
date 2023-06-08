Configuration
=============

Pyresample allows certain functionality to be controlled at a global level.
This allows users to quickly modify behavior at potentially very low levels
of pyresample without having to specify new arguments throughout their code.
Configuration is controlled through a central ``config`` object and allows
setting parameters in one of three ways:

1. Environment variable
2. YAML file
3. At runtime with ``pyresample.config``

This functionality is provided by the :doc:`donfig <donfig:configuration>`
library. The currently available settings are described below.
Each option is available from all three methods. If specified as an
environment variable or specified in the YAML file on disk, it must be set
**before** Pyresample is imported.

**YAML Configuration**

YAML files that include these parameters can be in any of the following
locations:

1. ``<python environment prefix>/etc/pyresample/pyresample.yaml``
2. ``<user_config_dir>/pyresample.yaml`` (see below)
3. ``~/.pyresample/pyresample.yaml``

The above ``user_config_dir`` is provided by the ``platformdirs`` package and
differs by operating system. Typical user config directories are:

* Mac OSX: ``~/Library/Preferences/pyresample``
* Unix/Linux: ``~/.config/pyresample``
* Windows: ``C:\\Users\\<username>\\AppData\\Local\\pytroll\\pyresample``

All YAML files found from the above paths will be merged into one
configuration object (accessed via ``pyresample.config``).
The YAML contents should be a simple mapping of configuration key to its
value. For example:

.. code-block:: yaml

    some_key: "some_value"

In some cases, keys may be grouped into sub-dictionaries:

.. code-block:: yaml

    features:
        future_geometries: true

Lastly, it is possible to specify an additional config path to the above
options by setting the environment variable ``PYRESAMPLE_CONFIG``. The file
specified with this environment variable will be added last after all of the
above paths have been merged together.

**At runtime**

After import, the values can be customized at runtime by doing:

.. code-block:: python

    import pyresample
    pyresamle.config.set(some_key="some_value")
    # ... normal pyresample code ...

Or for specific blocks of code:

.. code-block:: python

    import pyresample
    with pyresample.config.set(some_key="some_value):
        # ... some pyresample code ...
    # ... code using the original 'some_key' setting

Similarly, if you need to access one of the values you can
use the ``pyresample.config.get`` method.

Feature Flags
-------------

The below configuration options control whether certain features are made
available or used by default or overwrite existing behavior. In most cases
these are used for future changes to pyresample or experimental functionality
that may change later. These flags are all under the ``features``
sub-dictionary which requires some extra work to identify the substructure.
For example:

.. code-block:: python

    import pyresample
    pyresample.config.set({"features.future_geometries": True})

If using environment variables not the use of double underscores ``__`` in
parts of the variable name.

Future Geometries
^^^^^^^^^^^^^^^^^

* **Environment variable**: ``PYRESAMPLE_FEATURES__FUTURE_GEOMETRIES``
* **YAML/Config Key**: ``features: future_geometries``
* **Default**: False

Enable the use of future geometry objects (``AreaDefinition``,
``SwathDefinition``, etc) and overwrite any internal use of the old geometry
objects. This flag is meant to simplify the switch to future pyresample in
user code when utility methods like ``create_area_def`` are used. When enabled
the returned geometry instance will be of the future geometry class. These
classes can be accessed from:

.. code-block:: python

    from pyresample.future.geometry import AreaDefinition, SwathDefinition

Eventually these classes will be the default in Pyresample 2.0 and this flag
will have no effect.
