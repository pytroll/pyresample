#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2014-2021 Pyresample developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""The Boundary classes."""

from pyresample.boundary.simple_boundary import SimpleBoundary 
from pyresample.boundary.area_boundary import AreaBoundary, AreaDefBoundary 
from pyresample.boundary.geographic_boundary import GeographicBoundary 
from pyresample.boundary.projection_boundary import ProjectionBoundary 

__all__ = [ 
    "GeographicBoundary",
    "ProjectionBoundary",
    # Deprecated
    "SimpleBoundary",
    "AreaBoundary",
    "AreaDefBoundary",
]