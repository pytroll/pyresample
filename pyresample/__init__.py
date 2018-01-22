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

CHUNK_SIZE = os.getenv('PYTROLL_CHUNK_SIZE', 4096)

from pyresample.version import __version__
# Backwards compatibility
from pyresample import geometry
from pyresample import grid
from pyresample import image
from pyresample import kd_tree
from pyresample import utils
from pyresample import plot
# Easy access
from pyresample.geometry import (SwathDefinition,
                                 AreaDefinition,
                                 DynamicAreaDefinition)
from pyresample.utils import load_area
from pyresample.kd_tree import XArrayResamplerNN
from pyresample.plot import save_quicklook, area_def2basemap

__all__ = ['grid', 'image', 'kd_tree',
           'utils', 'plot', 'geo_filter', 'geometry', 'CHUNK_SIZE']


def get_capabilities():
    cap = {}

    try:
        from pykdtree.kdtree import KDTree
        cap['pykdtree'] = True
    except ImportError:
        cap['pykdtree'] = False

    try:
        import numexpr
        cap['numexpr'] = True
    except ImportError:
        cap['numexpr'] = False

    return cap
