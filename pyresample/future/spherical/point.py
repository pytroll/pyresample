#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013 - 2021 Pyresample developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Define SPoint and SMultiPoint class."""
import numpy as np

from pyresample.spherical import SCoordinate


def _plot(vertices, ax=None, **plot_kwargs):
    """Plot the SPoint/SMultiPoint using Cartopy.

    Assume vertices to in radians.
    """
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as ccrs
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Install cartopy to plot spherical geometries.")

    # Create figure if ax not provided
    ax_not_provided = False
    if ax is None:
        ax_not_provided = True
        proj_crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(subplot_kw=dict(projection=proj_crs))

    # Plot Points
    ax.scatter(x=np.rad2deg(vertices[:, 0]),
               y=np.rad2deg(vertices[:, 1]),
               **plot_kwargs)

    # Beautify plot by default
    if ax_not_provided:
        ax.stock_img()
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False

    return ax


def check_lon_validity(lon):
    """Check longitude validity."""
    if np.any(np.isinf(lon)):
        raise ValueError("Longitude values can not contain inf values.")


def check_lat_validity(lat):
    """Check latitude validity."""
    if np.any(np.isinf(lat)):
        raise ValueError("Latitude values can not contain inf values.")
    if np.any(np.logical_or(lat > np.pi / 2, lat < -np.pi / 2)):
        raise ValueError("Latitude values must range between [-pi/2, pi/2].")


class SPoint(SCoordinate):
    """Spherical Point.

    The ``lon`` and ``lat`` coordinates must be provided in radians.
    """

    def __init__(self, lon, lat):
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        if lon.size > 1 or lat.size > 1:
            raise ValueError("Use SMultiPoint to define multiple points.")
        check_lon_validity(lon)
        check_lat_validity(lat)

        super().__init__(lon, lat)

    @property
    def vertices(self):
        """Return SPoint vertices in a ndarray of shape [1,2]."""
        return np.array([self.lon, self.lat])[None, :]

    def plot(self, ax=None, **plot_kwargs):
        """Plot the SPoint using Cartopy."""
        ax = _plot(self.vertices, ax=ax, **plot_kwargs)
        return ax

    def to_shapely(self):
        """Convert the SPoint to a shapely Point (in lon/lat degrees)."""
        from shapely.geometry import Point
        point = Point(*np.rad2deg(self.vertices[0]))
        return point


class SMultiPoint(SCoordinate):
    """Create SPoint or SMultiPoint object."""

    def __new__(cls, lon, lat):
        """Create SPoint or SMultiPoint object."""
        lon = np.asarray(lon)
        lat = np.asarray(lat)

        # If a single point, define SPoint
        if lon.ndim == 0:
            return SPoint(lon, lat)  # TODO: TO BE DEFINED
        # Otherwise a SMultiPoint
        else:
            return super().__new__(cls)

    def __init__(self, lon, lat):
        super().__init__(lon, lat)

    @property
    def vertices(self):
        """Return SMultiPoint vertices in a ndarray of shape [n,2]."""
        return np.vstack((self.lon, self.lat)).T

    def __eq__(self, other):
        """Check equality."""
        return np.allclose(self.lon, other.lon) and np.allclose(self.lat, other.lat)

    def __str__(self):
        """Get simplified representation of lon/lat arrays in degrees."""
        vertices = np.rad2deg(np.vstack((self.lon, self.lat)).T)
        return str(vertices)

    def __repr__(self):
        """Get simplified representation of lon/lat arrays in degrees."""
        vertices = np.rad2deg(np.vstack((self.lon, self.lat)).T)
        return str(vertices)

    def plot(self, ax=None, **plot_kwargs):
        """Plot the SMultiPoint using Cartopy."""
        ax = _plot(self.vertices, ax=ax, **plot_kwargs)
        return ax

    def to_shapely(self):
        """Convert the SMultiPoint to a shapely MultiPoint (in lon/lat degrees)."""
        from shapely.geometry import MultiPoint
        point = MultiPoint(np.rad2deg(self.vertices))
        return point
