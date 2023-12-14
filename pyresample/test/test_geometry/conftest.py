#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010-2023 Pyresample developers
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
"""Define test AreaDefinitions."""
import pytest


@pytest.fixture(scope="session")
def stere_area(create_test_area):
    """Create basic polar-stereographic area definition."""
    proj_dict = {
        'a': '6378144.0',
        'b': '6356759.0',
        'lat_0': '50.00',
        'lat_ts': '50.00',
        'lon_0': '8.00',
        'proj': 'stere'
    }
    shape = (800, 800)
    area_extent = (-1370912.72, -909968.64000000001, 1029087.28, 1490031.3600000001)
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
        attrs={"name": 'areaD'},
    )


@pytest.fixture(scope="session")
def geos_src_area(create_test_area):
    """Create basic geostationary area definition."""
    shape = (3712, 3712)
    area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927, 5570248.477339745)
    proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                 'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
    )


@pytest.fixture(scope="session")
def laea_area(create_test_area):
    """Create basic LAEA area definition."""
    shape = (10, 10)
    area_extent = [1000000, 0, 1050000, 50000]
    proj_dict = {"proj": 'laea', 'lat_0': '60', 'lon_0': '0', 'a': '6371228.0', 'units': 'm'}
    return create_test_area(proj_dict, shape[0], shape[1], area_extent)


@pytest.fixture(scope="session")
def global_lonlat_antimeridian_centered_area(create_test_area):
    """Create global lonlat projection area centered on the -180 antimeridian."""
    shape = (4, 4)
    area_extent = (0, -90.0, 360, 90.0)
    proj_dict = '+proj=longlat +pm=180'
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
    )


@pytest.fixture(scope="session")
def global_platee_caree_area(create_test_area):
    """Create global platee projection area."""
    shape = (4, 4)
    area_extent = (-180.0, -90.0, 180.0, 90.0)
    proj_dict = 'EPSG:4326'
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
    )


@pytest.fixture(scope="session")
def global_platee_caree_minimum_area(create_test_area):
    """Create minimum size global platee caree projection area."""
    shape = (2, 2)
    area_extent = (-180.0, -90.0, 180.0, 90.0)
    proj_dict = 'EPSG:4326'
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
    )


@pytest.fixture(scope="session")
def local_platee_caree_area(create_test_area):
    """Create local platee caree projection area."""
    shape = (4, 4)
    area_extent = (100, 20, 120, 40)
    proj_dict = 'EPSG:4326'
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
    )


@pytest.fixture(scope="session")
def local_lonlat_antimeridian_centered_area(create_test_area):
    """Create local lonlat projection area centered on the -180 antimeridian."""
    shape = (4, 4)
    area_extent = (100, 20, 120, 40)
    proj_dict = '+proj=longlat +pm=180'
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
    )


@pytest.fixture(scope="session")
def local_meter_area(create_test_area):
    """Create local meter projection area."""
    shape = (2, 2)
    area_extent = (2_600_000.0, 1_050_000, 2_800_000.0, 1_170_000)
    proj_dict = 'EPSG:2056'
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
    )


@pytest.fixture(scope="session")
def south_pole_area(create_test_area):
    """Create projection area centered on south pole."""
    shape = (2, 2)
    area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
    proj_dict = {'proj': 'laea', 'lat_0': -90, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'}
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
    )


@pytest.fixture(scope="session")
def north_pole_area(create_test_area):
    """Create projection area centered on north pole."""
    shape = (2, 2)
    area_extent = (-5326849.0625, -5326849.0625, 5326849.0625, 5326849.0625)
    proj_dict = {'proj': 'laea', 'lat_0': 90, 'lon_0': 0, 'a': 6371228.0, 'units': 'm'}
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
    )


@pytest.fixture(scope="session")
def geos_fd_area(create_test_area):
    """Create full disc geostationary area definition."""
    shape = (100, 100)
    area_extent = (-5500000., -5500000., 5500000., 5500000.)
    proj_dict = {'a': 6378169.00, 'b': 6356583.80, 'h': 35785831.0,
                 'lon_0': 0, 'proj': 'geos', 'units': 'm'}
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
    )


@pytest.fixture(scope="session")
def geos_out_disk_area(create_test_area):
    """Create out of Earth diskc geostationary area definition."""
    shape = (10, 10)
    area_extent = (-5500000., -5500000., -5300000., -5300000.)
    proj_dict = {'a': 6378169.00, 'b': 6356583.80, 'h': 35785831.0,
                 'lon_0': 0, 'proj': 'geos', 'units': 'm'}
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
    )


@pytest.fixture(scope="session")
def geos_half_out_disk_area(create_test_area):
    """Create geostationary area definition with portion of boundary out of earth_disk."""
    shape = (100, 100)
    area_extent = (-5500000., -10000., 0, 10000.)
    proj_dict = {'a': 6378169.00, 'b': 6356583.80, 'h': 35785831.0,
                 'lon_0': 0, 'proj': 'geos', 'units': 'm'}
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
    )


@pytest.fixture(scope="session")
def geos_conus_area(create_test_area):
    """Create CONUS geostationary area definition (portion is out-of-Earth disk)."""
    shape = (30, 50)  # (3000, 5000) for GOES-R CONUS/PACUS
    proj_dict = {'h': 35786023, 'sweep': 'x', 'x_0': 0, 'y_0': 0,
                 'ellps': 'GRS80', 'no_defs': None, 'type': 'crs',
                 'lon_0': -75, 'proj': 'geos', 'units': 'm'}
    area_extent = (-3627271.29128, 1583173.65752, 1382771.92872, 4589199.58952)
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
    )


@pytest.fixture(scope="session")
def geos_mesoscale_area(create_test_area):
    """Create CONUS geostationary area definition."""
    shape = (10, 10)  # (1000, 1000) for GOES-R mesoscale
    proj_dict = {'h': 35786023, 'sweep': 'x', 'x_0': 0, 'y_0': 0,
                 'ellps': 'GRS80', 'no_defs': None, 'type': 'crs',
                 'lon_0': -75, 'proj': 'geos', 'units': 'm'}
    area_extent = (-501004.322, 3286588.35232, 501004.322, 4288596.99632)
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
    )


@pytest.fixture(scope="session")
def truncated_geos_area(create_test_area):
    """Create a truncated geostationary area (SEVIRI above 30° lat)."""
    proj_dict = {'a': '6378169', 'h': '35785831', 'lon_0': '9.5', 'no_defs': 'None', 'proj': 'geos',
                 'rf': '295.488065897014', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
    area_extent = (5567248.0742, 5570248.4773, -5570248.4773, 1393687.2705)
    shape = (1392, 3712)
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
    )


@pytest.fixture(scope="session")
def truncated_geos_area_in_space(create_test_area):
    """Create a geostationary area entirely out of the Earth disk !."""
    proj_dict = {'a': '6378169', 'h': '35785831', 'lon_0': '9.5', 'no_defs': 'None', 'proj': 'geos',
                 'rf': '295.488065897014', 'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'}
    area_extent = (5575000, 5575000, 5570000, 5570000)
    shape = (10, 10)
    return create_test_area(
        proj_dict,
        shape[0],
        shape[1],
        area_extent,
    )
