#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018 PyTroll developers
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
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Classes for geometry operations."""
from __future__ import annotations

from logging import getLogger
import numpy as np
import pyproj
import warnings

try:
    from xarray import DataArray
except ImportError:
    DataArray = np.ndarray

from pyresample.utils.proj4 import proj4_str_to_dict
import cartopy.crs as ccrs
import shapely.geometry as sgeom


logger = getLogger(__name__)

_GLOBE_PARAMS = {'datum': 'datum',
                 'ellps': 'ellipse',
                 'a': 'semimajor_axis',
                 'b': 'semiminor_axis',
                 'f': 'flattening',
                 'rf': 'inverse_flattening',
                 'towgs84': 'towgs84',
                 'nadgrids': 'nadgrids'}


def _globe_from_proj4(proj4_terms):
    """Create a `Globe` object from PROJ.4 parameters."""
    globe_terms = filter(lambda term: term[0] in _GLOBE_PARAMS,
                         proj4_terms.items())
    globe = ccrs.Globe(**{_GLOBE_PARAMS[name]: value for name, value in
                          globe_terms})
    return globe


class _PROJ4Projection(ccrs.Projection):

    def __init__(self, proj4_terms, globe=None, bounds=None):
        if 'EPSG' in proj4_terms.upper():
            warnings.warn('Converting EPSG projection to proj4 string, which is a potentially lossy transformation')
            proj4_terms = pyproj.Proj(proj4_terms).definition_string().strip()
        terms = proj4_str_to_dict(proj4_terms)
        globe = _globe_from_proj4(terms) if globe is None else globe

        other_terms = []
        for term in terms.items():
            if term[0] not in _GLOBE_PARAMS:
                other_terms.append(term)
        super(_PROJ4Projection, self).__init__(other_terms, globe)

        self.bounds = bounds

    def __repr__(self):
        return '_PROJ4Projection({})'.format(self.proj4_init)

    @property
    def boundary(self):
        x0, x1, y0, y1 = self.bounds
        return sgeom.LineString([(x0, y0), (x0, y1), (x1, y1), (x1, y0),
                                 (x0, y0)])

    @property
    def x_limits(self):
        x0, x1, y0, y1 = self.bounds
        return (x0, x1)

    @property
    def y_limits(self):
        x0, x1, y0, y1 = self.bounds
        return (y0, y1)

    @property
    def threshold(self):
        x0, x1, y0, y1 = self.bounds
        return min(abs(x1 - x0), abs(y1 - y0)) / 100.


class _Projection(ccrs.Projection):
    def __init__(self,
                 crs: pyproj.CRS,
                 bounds: list[float, float, float, float] = None,
                 transform_bounds: bool = False):
        # NOTE: Skip the base cartopy Projection class __init__
        super(ccrs.Projection, self).__init__(crs)
        if bounds is None and self.area_of_use is not None:
            bounds = (
                self.area_of_use.west,
                self.area_of_use.east,
                self.area_of_use.south,
                self.area_of_use.north,
            )
            transform_bounds = True

        self.bounds = bounds
        if bounds is not None and transform_bounds:
            # Convert lat/lon bounds to projected bounds.
            # Geographic area of the entire dataset referenced to WGS 84
            # NB. We can't use a polygon transform at this stage because
            # that relies on the existence of the map boundary... the very
            # thing we're trying to work out! ;-)
            x0, x1, y0, y1 = bounds
            lons = np.array([x0, x0, x1, x1])
            lats = np.array([y0, y1, y1, y0])
            points = self.transform_points(self.as_geodetic(), lons, lats)
            x = points[:, 0]
            y = points[:, 1]
            self.bounds = (x.min(), x.max(), y.min(), y.max())
        if self.bounds is not None:
            x0, x1, y0, y1 = self.bounds
            self.threshold = min(abs(x1 - x0), abs(y1 - y0)) / 100.


def _lesser_from_proj(proj4_terms, globe=None, bounds=None):
    """Not-as-good version of cartopy's 'from_proj' function.

    The user doesn't have a newer version of Cartopy so there is no
    `from_proj` function to use which does a fancier job of creating CRS
    objects from PROJ.4 strings than this does.
    """
    return _PROJ4Projection(proj4_terms, globe=globe, bounds=bounds)


from_proj = _lesser_from_proj

if issubclass(ccrs.Projection, pyproj.CRS):
    Projection = _Projection
