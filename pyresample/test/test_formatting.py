#!/usr/bin/env python

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
"""Test html formatting."""

import pyresample
from pyresample._formatting_html import area_repr, swath_area_attrs_section

from .test_geometry.test_area import stere_area  # noqa F401
from .test_geometry.test_swath import _gen_swath_def_numpy, _gen_swath_def_xarray_dask


def test_area_def_cartopy_missing(monkeypatch, stere_area):  # noqa F811
    """Test missing cartopy installation."""
    with monkeypatch.context() as m:
        m.setattr(pyresample._formatting_html, "cart", False)

        area = stere_area
        assert "Note: If cartopy is installed a display of the area can be seen here" in area._repr_html_()


def test_area_def_cartopy_installed(stere_area):  # noqa F811
    """Test cartopy installed."""
    area = stere_area
    assert "Note: If cartopy is installed a display of the area can be seen here" not in area._repr_html_()


def test_area_repr_w_static_files(stere_area):  # noqa F811
    """Test area representation with static files (css/icons) included."""
    area_def = stere_area
    res = area_repr(area_def)
    assert "<style>" in res


def test_area_repr_wo_static_files(stere_area):  # noqa F811
    """Test area representation without static files (css/icons) included."""
    area_def = stere_area
    res = area_repr(area_def, include_static_files=False)
    assert "<style>" not in res


def test_swath_area_attrs_section_w_numpy(create_test_swath):
    """Test SwathDefinition attrs section with numpy lons/lats."""
    swath_def = _gen_swath_def_numpy(create_test_swath)
    res = swath_area_attrs_section(swath_def)
    assert "class=\'xr-text-repr-fallback\'" not in res


def test_swath_area_attrs_section_w_xarray(create_test_swath):
    """Test SwathDefinition attrs section with numpy lons/lats."""
    swath_def = _gen_swath_def_xarray_dask(create_test_swath)
    res = swath_area_attrs_section(swath_def)
    assert "class=\'xr-text-repr-fallback\'" in res
