#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2010-2022 Pyresample developers
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
"""Test cartopy plotting utilities."""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pytest
from shapely.geometry import Polygon

from pyresample.visualization.geometries import (
    _add_map_background,
    _check_subplot_kw,
    _initialize_plot,
    plot_geometries,
)


class TestPlotFunctions:
    """Test suite for the provided plotting functions."""

    def test_add_map_background(self):
        """Test adding a map background to an axis."""
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        result_ax = _add_map_background(ax)
        assert isinstance(result_ax, plt.Axes)

    def test_check_subplot_kw_valid(self):
        """Test _check_subplot_kw with valid input."""
        valid_kw = {'projection': ccrs.PlateCarree()}
        assert _check_subplot_kw(valid_kw) == valid_kw

    def test_check_subplot_kw_none(self):
        """Test _check_subplot_kw with None input."""
        assert 'projection' in _check_subplot_kw(None)

    def test_check_subplot_kw_invalid(self):
        """Test _check_subplot_kw with invalid input."""
        with pytest.raises(TypeError):
            _check_subplot_kw("invalid")

        with pytest.raises(TypeError):
            _check_subplot_kw(2)

        with pytest.raises(TypeError):
            _check_subplot_kw([2])

        with pytest.raises(ValueError):
            _check_subplot_kw({})

    def test_initialize_plot_with_ax(self):
        """Test _initialize_plot with an existing ax."""
        fig, ax = plt.subplots()
        _, result_ax, initialized_here = _initialize_plot(ax=ax)
        assert result_ax == ax
        assert not initialized_here

    @pytest.mark.parametrize("ax_provided", [True, False])
    def test_plot_geometries(self, ax_provided):
        """Test plot_geometries function returns the correct type based on ax_provided."""
        import cartopy
        vertices1 = [(0, 0), (0, 1), (1, 0)]
        vertices2 = [(0, 0), (0, 2), (2, 0)]
        geometries = [Polygon(vertices1), Polygon(vertices2)]
        crs = ccrs.PlateCarree()
        ax = plt.axes(projection=crs) if ax_provided else None
        result = plot_geometries(geometries, crs, ax=ax)
        assert isinstance(result, cartopy.mpl.feature_artist.FeatureArtist)
