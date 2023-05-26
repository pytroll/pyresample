#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010-2021 Pyresample developers
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Pyresample package for geographic data resampling and related utilities."""

import os

# Backwards compatibility
from pyresample import geometry  # noqa
from pyresample import grid  # noqa
from pyresample import image  # noqa
from pyresample import kd_tree  # noqa
from pyresample import plot  # noqa
from pyresample import utils  # noqa
from pyresample.area_config import (  # noqa
    convert_def_to_yaml,
    create_area_def,
    get_area_def,
    load_area,
    parse_area_file,
)

# Easy access
from pyresample.geometry import AreaDefinition  # noqa
from pyresample.geometry import DynamicAreaDefinition  # noqa
from pyresample.geometry import SwathDefinition  # noqa
from pyresample.kd_tree import XArrayResamplerNN  # noqa
from pyresample.plot import area_def2basemap, save_quicklook  # noqa

from pyresample._config import config  # isort: skip <- import first so everything below has access

LegacyAreaDefinition = AreaDefinition
LegacySwathDefinition = SwathDefinition
if config.get("features.future_geometries"):
    from pyresample.future.geometry import AreaDefinition, SwathDefinition

from .version import get_versions  # noqa

CHUNK_SIZE = int(os.getenv('PYTROLL_CHUNK_SIZE', 4096))

__all__ = ['grid', 'image', 'kd_tree', 'utils', 'plot', 'geo_filter', 'geometry', 'CHUNK_SIZE',
           'load_area', 'create_area_def', 'get_area_def', 'parse_area_file', 'convert_def_to_yaml']

__version__ = get_versions()['version']
del get_versions
