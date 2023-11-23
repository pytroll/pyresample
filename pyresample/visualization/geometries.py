#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010-2020 Pyresample developers
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
"""Define how to plot a shapely geometry."""


def _add_map_background(ax):
    """Add cartopy map background."""
    ax.stock_img()
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    return ax


def _check_subplot_kw(subplot_kw):
    """Check subplot_kw arguments."""
    import cartopy.crs as ccrs

    if subplot_kw is None:
        subplot_kw = dict(projection=ccrs.PlateCarree())
    if not isinstance(subplot_kw, dict):
        raise TypeError("'subplot_kw' must be a dictionary.'")
    if "projection" not in subplot_kw:
        raise ValueError("Specify a cartopy 'projection' in subplot_kw.")
    return subplot_kw


def _initialize_plot(ax=None, subplot_kw=None):
    """Initialize plot."""
    import matplotlib.pyplot as plt

    if ax is None:
        subplot_kw = _check_subplot_kw(subplot_kw)
        fig, ax = plt.subplots(subplot_kw=subplot_kw)
        return fig, ax, True
    else:
        return None, ax, False


def plot_geometries(geometries, crs, ax=None, subplot_kw=None, **kwargs):
    """Plot geometries in cartopy."""
    # Create figure if ax not provided
    fig, ax, initialized_here = _initialize_plot(ax=ax, subplot_kw=subplot_kw)
    # Add map background if ax not provided as input
    if initialized_here:
        ax = _add_map_background(ax)
    # Add  geometries
    ax.add_geometries(geometries, crs=crs, **kwargs)
    # Return Figure / Axis
    if initialized_here:
        return fig
    else:
        return ax
