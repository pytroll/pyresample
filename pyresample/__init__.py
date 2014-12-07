#pyresample, Resampling of remote sensing image data in python
# 
#Copyright (C) 2010  Esben S. Nielsen
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import

from . import grid
from . import image
from . import kd_tree
from . import utils
from . import version
from . import plot

__version__ = version.__version__

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
