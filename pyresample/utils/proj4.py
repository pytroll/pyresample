#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019-2021 Pyresample developers
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
"""Utilities for working with projection parameters."""
import contextlib
import math
import warnings
from collections import OrderedDict

import numpy as np
from pyproj import CRS
from pyproj import Transformer as PROJTransformer


def convert_proj_floats(proj_pairs):
    """Convert PROJ.4 parameters to floats if possible."""
    proj_dict = OrderedDict()
    for x in proj_pairs:
        if len(x) == 1 or x[1] is True:
            proj_dict[x[0]] = True
            continue

        try:
            proj_dict[x[0]] = float(x[1])
        except ValueError:
            proj_dict[x[0]] = x[1]

    return proj_dict


def proj4_str_to_dict(proj4_str):
    """Convert PROJ.4 compatible string definition to dict.

    EPSG codes should be provided as "EPSG:XXXX" where "XXXX"
    is the EPSG number code. It can also be provided as
    ``"+init=EPSG:XXXX"`` as long as the underlying PROJ library
    supports it (deprecated in PROJ 6.0+).

    Note: Key only parameters will be assigned a value of `True`.
    """
    # convert EPSG codes to equivalent PROJ4 string definition
    crs = CRS(proj4_str)
    return crs.to_dict()


def proj4_dict_to_str(proj4_dict, sort=False):
    """Convert a dictionary of PROJ.4 parameters to a valid PROJ.4 string."""
    items = proj4_dict.items()
    if sort:
        items = sorted(items)
    params = []
    for key, val in items:
        key = str(key) if key.startswith('+') else '+' + str(key)
        if key in ['+no_defs', '+no_off', '+no_rot'] or val is None:
            param = key
        else:
            param = '{}={}'.format(key, val)
        params.append(param)
    return ' '.join(params)


def proj4_radius_parameters(proj4_dict):
    """Calculate 'a' and 'b' radius parameters.

    Arguments:
        proj4_dict (str or dict): PROJ.4 parameters

    Returns:
        a (float), b (float): equatorial and polar radius
    """
    crs = proj4_dict
    if not isinstance(crs, CRS):
        crs = CRS(crs)
    a = crs.ellipsoid.semi_major_metre
    b = crs.ellipsoid.semi_minor_metre
    if not math.isnan(b):
        return a, b


def get_geostationary_height(geos_area_crs):
    """Get the height parameter from a geostationary CRS."""
    params = geos_area_crs.coordinate_operation.params
    h_param = [p for p in params if 'satellite height' in p.name.lower()][0]
    return h_param.value


def _transform_dask_chunk(x, y, crs_from, crs_to, kwargs, transform_kwargs):
    crs_from = CRS(crs_from)
    crs_to = CRS(crs_to)
    transformer = PROJTransformer.from_crs(crs_from, crs_to, **kwargs)
    return np.stack(transformer.transform(x, y, **transform_kwargs), axis=-1)


class DaskFriendlyTransformer:
    """Wrapper around the pyproj Transformer class that uses dask.

    If the provided arrays are not dask arrays, they are converted to numpy
    arrays and pyproj will be called directly (dask is not used).

    """

    def __init__(self, src_crs, dst_crs, **kwargs):
        """Initialize the transformer with CRS objects.

        This method should not be used directly, just like pyproj.Transformer
        should not be created directly.

        """
        self.src_crs = src_crs
        self.dst_crs = dst_crs
        self.kwargs = kwargs

    @classmethod
    def from_crs(cls, crs_from, crs_to, **kwargs):
        """Create transformer object from two CRS objects."""
        return cls(crs_from, crs_to, **kwargs)

    def transform(self, x, y, **kwargs):
        """Transform coordinates."""
        import dask.array as da
        crs_from = self.src_crs
        crs_to = self.dst_crs

        if not hasattr(x, "compute"):
            x = np.asarray(x)
            y = np.asarray(y)
            return self._transform_numpy(x, y, **kwargs)

        # CRS objects aren't thread-safe until pyproj 3.1+
        # convert to WKT strings to be safe
        result = da.map_blocks(_transform_dask_chunk, x, y,
                               crs_from.to_wkt(), crs_to.to_wkt(),
                               dtype=x.dtype, chunks=x.chunks + ((2,),),
                               meta=np.array((), dtype=x.dtype),
                               kwargs=self.kwargs,
                               transform_kwargs=kwargs,
                               new_axis=x.ndim)
        x = result[..., 0]
        y = result[..., 1]
        return x, y

    def _transform_numpy(self, x, y, **kwargs):
        transformer = PROJTransformer.from_crs(self.src_crs, self.dst_crs, **self.kwargs)
        return transformer.transform(x, y, **kwargs)


@contextlib.contextmanager
def ignore_pyproj_proj_warnings():
    """Wrap operations that we know will produce a PROJ.4 precision warning.

    Only to be used internally to Pyresample when we have no other choice but
    to use PROJ.4 strings/dicts. For example, serialization to YAML or other
    human-readable formats or testing the methods that produce the PROJ.4
    versions of the CRS.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "You will likely lose important projection information",
            UserWarning,
        )
        yield


def get_geodetic_crs_with_no_datum_shift(crs: CRS) -> CRS:
    """Get the geodetic CRS for the provided CRS but with no prime meridian shift."""
    gcrs = crs.geodetic_crs
    if gcrs.prime_meridian.longitude == 0:
        return gcrs
    with ignore_pyproj_proj_warnings():
        gcrs_dict = gcrs.to_dict()
    gcrs_dict.pop("pm", None)
    gcrs_pm0 = CRS.from_dict(gcrs_dict)
    return gcrs_pm0
