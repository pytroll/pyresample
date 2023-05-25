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
from pyresample.geometry import AreaDefinition


def test_area_def_cartopy_missing(monkeypatch):
    """Test missing cartopy installation."""
    projection = {'a': '6378144.0',
                  'b': '6356759.0',
                  'lat_0': '50.00',
                  'lat_ts': '50.00',
                  'lon_0': '8.00',
                  'proj': 'stere'}
    proj_id = "test"
    width = 800
    height = 800
    area_extent = (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001)
    area_id = "areaD"
    description = "test"

    with monkeypatch.context() as m:
        m.setattr(pyresample._formatting_html, "cart", False)

        area = AreaDefinition(area_id, description, proj_id, projection, width, height, area_extent)
        assert "Note: If cartopy is installed a display of the area can be seen here" in area._repr_html_()


def test_area_def_cartopy_installed():
    """Test cartopy installed."""
    projection = {'a': '6378144.0',
                  'b': '6356759.0',
                  'lat_0': '50.00',
                  'lat_ts': '50.00',
                  'lon_0': '8.00',
                  'proj': 'stere'}
    proj_id = "test"
    width = 800
    height = 800
    area_extent = (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001)
    area_id = "areaD"
    description = "test"

    area = AreaDefinition(area_id, description, proj_id, projection, width, height, area_extent)
    assert "Note: If cartopy is installed a display of the area can be seen here" not in area._repr_html_()
