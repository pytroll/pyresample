#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 David Hoese
# Author(s):
#   David Hoese <david.hoese@ssec.wisc.edu>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Utilities for testing.

This mostly takes from astropy's method for checking warnings during tests.
"""
import sys
import types
import warnings

import numpy as np

try:
    from pyproj import CRS
except ImportError:
    CRS = None

_deprecations_as_exceptions = False
_include_astropy_deprecations = False
AstropyDeprecationWarning = None
AstropyPendingDeprecationWarning = None


def treat_deprecations_as_exceptions():
    """
    Turn all DeprecationWarnings (which indicate deprecated uses of
    Python itself or Numpy, but not within Astropy, where we use our
    own deprecation warning class) into exceptions so that we find
    out about them early.

    This completely resets the warning filters and any "already seen"
    warning state.
    """
    # First, totally reset the warning state
    for module in sys.modules.values():
        # We don't want to deal with six.MovedModules, only "real"
        # modules.
        if (isinstance(module, types.ModuleType) and
                hasattr(module, '__warningregistry__')):
            del module.__warningregistry__

    if not _deprecations_as_exceptions:
        return

    warnings.resetwarnings()

    # Hide the next couple of DeprecationWarnings
    warnings.simplefilter('ignore', DeprecationWarning)
    # Here's the wrinkle: a couple of our third-party dependencies
    # (py.test and scipy) are still using deprecated features
    # themselves, and we'd like to ignore those.  Fortunately, those
    # show up only at import time, so if we import those things *now*,
    # before we turn the warnings into exceptions, we're golden.
    try:
        # A deprecated stdlib module used by py.test
        import compiler  # noqa
    except ImportError:
        pass

    try:
        import scipy  # noqa
    except ImportError:
        pass

    # Now, start over again with the warning filters
    warnings.resetwarnings()
    # Now, turn DeprecationWarnings into exceptions
    warnings.filterwarnings("error", ".*", DeprecationWarning)

    # Only turn astropy deprecation warnings into exceptions if requested
    if _include_astropy_deprecations:
        warnings.filterwarnings("error", ".*", AstropyDeprecationWarning)
        warnings.filterwarnings("error", ".*", AstropyPendingDeprecationWarning)

    # py.test reads files with the 'U' flag, which is now
    # deprecated in Python 3.4.
    warnings.filterwarnings(
        "ignore",
        r"'U' mode is deprecated",
        DeprecationWarning)

    # BeautifulSoup4 triggers a DeprecationWarning in stdlib's
    # html module.x
    warnings.filterwarnings(
        "ignore",
        r"The strict argument and mode are deprecated\.",
        DeprecationWarning)
    warnings.filterwarnings(
        "ignore",
        r"The value of convert_charrefs will become True in 3\.5\. "
        r"You are encouraged to set the value explicitly\.",
        DeprecationWarning)
    # Filter out pyresample's deprecation warnings.
    warnings.filterwarnings(
        "ignore",
        r"This module will be removed in pyresample 2\.0\, please use the"
        r"\`pyresample.spherical\` module functions and class instead\.",
        DeprecationWarning)

    if sys.version_info[:2] >= (3, 5):
        # py.test raises this warning on Python 3.5.
        # This can be removed when fixed in py.test.
        # See https://github.com/pytest-dev/pytest/pull/1009
        warnings.filterwarnings(
            "ignore",
            r"inspect\.getargspec\(\) is deprecated, use "
            r"inspect\.signature\(\) instead",
            DeprecationWarning)


class catch_warnings(warnings.catch_warnings):
    """
    A high-powered version of warnings.catch_warnings to use for testing
    and to make sure that there is no dependence on the order in which
    the tests are run.

    This completely blitzes any memory of any warnings that have
    appeared before so that all warnings will be caught and displayed.

    ``*args`` is a set of warning classes to collect.  If no arguments are
    provided, all warnings are collected.

    Use as follows::

        with catch_warnings(MyCustomWarning) as w:
            do.something.bad()
        assert len(w) > 0
    """
    def __init__(self, *classes):
        super(catch_warnings, self).__init__(record=True)
        self.classes = classes

    def __enter__(self):
        warning_list = super(catch_warnings, self).__enter__()
        treat_deprecations_as_exceptions()
        if len(self.classes) == 0:
            warnings.simplefilter('always')
        else:
            warnings.simplefilter('ignore')
            for cls in self.classes:
                warnings.simplefilter('always', cls)
        return warning_list

    def __exit__(self, type, value, traceback):
        treat_deprecations_as_exceptions()


def create_test_longitude(start, stop, shape, twist_factor=0.0, dtype=np.float32):
    if start > stop:
        stop += 360.0

    lon_row = np.linspace(start, stop, num=shape[1]).astype(dtype)
    twist_array = np.arange(shape[0]).reshape((shape[0], 1)) * twist_factor
    lon_array = np.repeat([lon_row], shape[0], axis=0)
    lon_array += twist_array

    if stop > 360.0:
        lon_array[lon_array > 360.0] -= 360
    return lon_array


def create_test_latitude(start, stop, shape, twist_factor=0.0, dtype=np.float32):
    lat_col = np.linspace(start, stop, num=shape[0]).astype(dtype).reshape((shape[0], 1))
    twist_array = np.arange(shape[1]) * twist_factor
    lat_array = np.repeat(lat_col, shape[1], axis=1)
    lat_array += twist_array
    return lat_array


class CustomScheduler(object):
    """Scheduler raising an exception if data are computed too many times."""

    def __init__(self, max_computes=1):
        """Set starting and maximum compute counts."""
        self.max_computes = max_computes
        self.total_computes = 0

    def __call__(self, dsk, keys, **kwargs):
        """Compute dask task and keep track of number of times we do so."""
        import dask
        self.total_computes += 1
        if self.total_computes > self.max_computes:
            raise RuntimeError("Too many dask computations were scheduled: "
                               "{}".format(self.total_computes))
        return dask.get(dsk, keys, **kwargs)


def friendly_crs_equal(expected, actual, keys=None, use_obj=True, use_wkt=True):
    """Test if two projection definitions are equal.

    The main purpose of this function is to help manage differences
    between pyproj versions. Depending on the version installed and used
    pyresample may provide a different `proj_dict` or other similar
    CRS definition.

    Args:
        expected (dict, str, pyproj.crs.CRS): Expected CRS definition as
            a PROJ dictionary or string or CRS object.
        actual (dict, str, pyproj.crs.CRS): Actual CRS definition
        keys (list): Specific PROJ parameters to look for. Only takes effect
            if `use_obj` is `False`.
        use_obj (bool): Use pyproj's CRS object to test equivalence. Default
            is True.
        use_wkt (bool): Increase likely hood of making CRS objects equal by
            converting WellKnownText before converting to the final CRS
            object. Requires `use_obj`. Defaults to True.

    """
    if CRS is not None and use_obj:
        if hasattr(expected, 'crs'):
            expected = expected.crs
        if hasattr(actual, 'crs'):
            actual = actual.crs
        expected_crs = CRS(expected)
        actual_crs = CRS(actual)
        if use_wkt:
            expected_crs = CRS(expected_crs.to_wkt())
            actual_crs = CRS(actual_crs.to_wkt())
        return expected_crs == actual_crs
    raise NotImplementedError("""TODO""")
