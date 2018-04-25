# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2010, 2015  Esben S. Nielsen
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

"""Reduce data sets based on geographical information"""

from __future__ import absolute_import

import numpy as np

# Earth radius
R = 6370997.0


def swath_from_cartesian_grid(cart_grid, lons, lats, data,
                              radius_of_influence):
    """Makes coarse data reduction of swath data by comparison with
    cartesian grid

    Parameters
    ----------
    chart_grid : numpy array
        Grid of area cartesian coordinates
    lons : numpy array
        Swath lons
    lats : numpy array
        Swath lats
    data : numpy array
        Swath data
    radius_of_influence : float
        Cut off distance in meters

    Returns
    -------
    (lons, lats, data) : list of numpy arrays
        Reduced swath data and coordinate set
    """

    valid_index = get_valid_index_from_cartesian_grid(cart_grid, lons, lats,
                                                      radius_of_influence)

    lons = lons[valid_index]
    lats = lats[valid_index]
    data = data[valid_index]

    return lons, lats, data


def get_valid_index_from_cartesian_grid(cart_grid, lons, lats,
                                        radius_of_influence):
    """Calculates relevant data indices using coarse data reduction of swath
    data by comparison with cartesian grid

    Parameters
    ----------
    chart_grid : numpy array
        Grid of area cartesian coordinates
    lons : numpy array
        Swath lons
    lats : numpy array
        Swath lats
    data : numpy array
        Swath data
    radius_of_influence : float
        Cut off distance in meters

    Returns
    -------
    valid_index : numpy array
        Boolean array of same size as lons and lats indicating relevant indices
    """

    def _get_lons(x, y):
        return np.rad2deg(np.arccos(x / np.sqrt(x ** 2 + y ** 2))) * np.sign(y)

    def _get_lats(z):
        return 90 - np.rad2deg(np.arccos(z / R))

    # Get sides of target grid and transform to lon lats
    lons_side1 = _get_lons(cart_grid[0, :, 0], cart_grid[0, :, 1])
    lons_side2 = _get_lons(cart_grid[:, -1, 0], cart_grid[:, -1, 1])
    lons_side3 = _get_lons(cart_grid[-1, ::-1, 0], cart_grid[-1, ::-1, 1])
    lons_side4 = _get_lons(cart_grid[::-1, 0, 0], cart_grid[::-1, 0, 1])

    lats_side1 = _get_lats(cart_grid[0, :, 2])
    lats_side2 = _get_lats(cart_grid[:, -1, 2])
    lats_side3 = _get_lats(cart_grid[-1, ::-1, 2])
    lats_side4 = _get_lats(cart_grid[::-1, 0, 2])

    valid_index = _get_valid_index(lons_side1, lons_side2, lons_side3, lons_side4,
                                   lats_side1, lats_side2, lats_side3, lats_side4,
                                   lons, lats, radius_of_influence)

    return valid_index


def swath_from_lonlat_grid(grid_lons, grid_lats, lons, lats, data,
                           radius_of_influence):
    """Makes coarse data reduction of swath data by comparison with
    lon lat grid

    Parameters
    ----------
    grid_lons : numpy array
        Grid of area lons
    grid_lats : numpy array
        Grid of area lats
    lons : numpy array
        Swath lons
    lats : numpy array
        Swath lats
    data : numpy array
        Swath data
    radius_of_influence : float
        Cut off distance in meters

    Returns
    -------
    (lons, lats, data) : list of numpy arrays
        Reduced swath data and coordinate set
    """

    valid_index = get_valid_index_from_lonlat_grid(
        grid_lons, grid_lats, lons, lats, radius_of_influence)

    lons = lons[valid_index]
    lats = lats[valid_index]
    data = data[valid_index]

    return lons, lats, data


def swath_from_lonlat_boundaries(boundary_lons, boundary_lats, lons, lats, data,
                                 radius_of_influence):
    """Makes coarse data reduction of swath data by comparison with
    lon lat boundary

    Parameters
    ----------
    boundary_lons : numpy array
        Grid of area lons
    boundary_lats : numpy array
        Grid of area lats
    lons : numpy array
        Swath lons
    lats : numpy array
        Swath lats
    data : numpy array
        Swath data
    radius_of_influence : float
        Cut off distance in meters

    Returns
    -------
    (lons, lats, data) : list of numpy arrays
        Reduced swath data and coordinate set
    """

    valid_index = get_valid_index_from_lonlat_boundaries(boundary_lons,
                                                         boundary_lats, lons, lats, radius_of_influence)

    lons = lons[valid_index]
    lats = lats[valid_index]
    data = data[valid_index]

    return lons, lats, data


def get_valid_index_from_lonlat_grid(grid_lons, grid_lats, lons, lats, radius_of_influence):
    """Calculates relevant data indices using coarse data reduction of swath
    data by comparison with lon lat grid

    Parameters
    ----------
    chart_grid : numpy array
        Grid of area cartesian coordinates
    lons : numpy array
        Swath lons
    lats : numpy array
        Swath lats
    data : numpy array
        Swath data
    radius_of_influence : float
        Cut off distance in meters

    Returns
    -------
    valid_index : numpy array
        Boolean array of same size as lon and lat indicating relevant indices
    """

    # Get sides of target grid
    lons_side1 = grid_lons[0, :]
    lons_side2 = grid_lons[:, -1]
    lons_side3 = grid_lons[-1, ::-1]
    lons_side4 = grid_lons[::-1, 0]

    lats_side1 = grid_lats[0, :]
    lats_side2 = grid_lats[:, -1]
    lats_side3 = grid_lats[-1, :]
    lats_side4 = grid_lats[:, 0]

    valid_index = _get_valid_index(lons_side1, lons_side2, lons_side3, lons_side4,
                                   lats_side1, lats_side2, lats_side3, lats_side4,
                                   lons, lats, radius_of_influence)

    return valid_index


def get_valid_index_from_lonlat_boundaries(boundary_lons, boundary_lats, lons, lats, radius_of_influence):
    """Find relevant indices from grid boundaries using the
    winding number theorem"""

    valid_index = _get_valid_index(boundary_lons.side1, boundary_lons.side2,
                                   boundary_lons.side3, boundary_lons.side4,
                                   boundary_lats.side1, boundary_lats.side2,
                                   boundary_lats.side3, boundary_lats.side4,
                                   lons, lats, radius_of_influence)

    return valid_index


def _get_valid_index(lons_side1, lons_side2, lons_side3, lons_side4,
                     lats_side1, lats_side2, lats_side3, lats_side4,
                     lons, lats, radius_of_influence):
    """Find relevant indices from grid boundaries using the
    winding number theorem"""

    # Coarse reduction of data based on extrema analysis of the boundary
    # lon lat values of the target grid
    illegal_lons = (((lons_side1 < -180) | (lons_side1 > 180)).any() or
                    ((lons_side2 < -180) | (lons_side2 > 180)).any() or
                    ((lons_side3 < -180) | (lons_side3 > 180)).any() or
                    ((lons_side4 < -180) | (lons_side4 > 180)).any())

    illegal_lats = (((lats_side1 < -90) | (lats_side1 > 90)).any() or
                    ((lats_side2 < -90) | (lats_side2 > 90)).any() or
                    ((lats_side3 < -90) | (lats_side3 > 90)).any() or
                    ((lats_side4 < -90) | (lats_side4 > 90)).any())

    if illegal_lons or illegal_lats:
        # Grid boundaries are not safe to operate on
        return np.ones(lons.size, dtype=np.bool)

    # Find sum angle sum of grid boundary
    angle_sum = 0
    for side in (lons_side1, lons_side2, lons_side3, lons_side4):
        prev = None
        side_sum = 0
        for lon in side:
            if prev:
                delta = lon - prev
                if abs(delta) > 180:
                    delta = (abs(delta) - 360) * (delta // abs(delta))
                angle_sum += delta
                side_sum += delta
            prev = lon

    # Buffer min and max lon and lat of interest with radius of interest
    lat_min = min(lats_side1.min(), lats_side2.min(), lats_side3.min(),
                  lats_side4.min())
    lat_min_buffered = lat_min - np.degrees(float(radius_of_influence) / R)
    lat_max = max(lats_side1.max(), lats_side2.max(), lats_side3.max(),
                  lats_side4.max())
    lat_max_buffered = lat_max + np.degrees(float(radius_of_influence) / R)

    max_angle_s2 = max(abs(lats_side2.max()), abs(lats_side2.min()))
    max_angle_s4 = max(abs(lats_side4.max()), abs(lats_side4.min()))
    lon_min_buffered = (lons_side4.min() -
                        np.degrees(float(radius_of_influence) /
                                   (np.sin(np.radians(max_angle_s4)) * R)))

    lon_max_buffered = (lons_side2.max() +
                        np.degrees(float(radius_of_influence) /
                                   (np.sin(np.radians(max_angle_s2)) * R)))

    # From the winding number theorem follows:
    # angle_sum possiblilities:
    # -360: area covers north pole
    # 360: area covers south pole
    #   0: area covers no poles
    # else: area covers both poles
    if round(angle_sum) == -360:
        # Covers NP
        valid_index = (lats >= lat_min_buffered)
    elif round(angle_sum) == 360:
        # Covers SP
        valid_index = (lats <= lat_max_buffered)
    elif round(angle_sum) == 0:
        # Covers no poles
        valid_lats = (lats >= lat_min_buffered) * (lats <= lat_max_buffered)

        if lons_side2.min() > lons_side4.max():
            # No date line crossing
            valid_lons = (lons >= lon_min_buffered) * \
                (lons <= lon_max_buffered)
        else:
            # Date line crossing
            seg1 = (lons >= lon_min_buffered) * (lons <= 180)
            seg2 = (lons <= lon_max_buffered) * (lons >= -180)
            valid_lons = seg1 + seg2

        valid_index = valid_lats * valid_lons
    else:
        # Covers both poles don't reduce
        valid_index = np.ones(lons.size, dtype=np.bool)

    return valid_index
