# pyresample, Resampling of remote sensing image data in python
#
# Copyright (C) 2010, 2014, 2015  Esben S. Nielsen
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

import os

CHUNK_SIZE = int(os.getenv('PYTROLL_CHUNK_SIZE', 4096))

# Backwards compatibility
from pyresample import geometry  # noqa
from pyresample import grid  # noqa
from pyresample import image  # noqa
from pyresample import kd_tree  # noqa
from pyresample import utils  # noqa
from pyresample import plot  # noqa
# Easy access
from pyresample.geometry import (SwathDefinition,  # noqa
                                 AreaDefinition,  # noqa
                                 DynamicAreaDefinition)  # noqa
from pyresample.area_config import load_area, create_area_def, get_area_def, \
                                   parse_area_file, convert_def_to_yaml  # noqa
from pyresample.kd_tree import XArrayResamplerNN  # noqa
from pyresample.plot import save_quicklook, area_def2basemap  # noqa
from .version import get_versions  # noqa

__all__ = ['grid', 'image', 'kd_tree', 'utils', 'plot', 'geo_filter', 'geometry', 'CHUNK_SIZE',
           'load_area', 'create_area_def', 'get_area_def', 'parse_area_file', 'convert_def_to_yaml']

__version__ = get_versions()['version']
del get_versions
