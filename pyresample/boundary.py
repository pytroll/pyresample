#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014, 2015, 2017 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""The Boundary classes.
"""


import logging
import logging.handlers

import numpy as np

from pyresample.spherical import SphPolygon

logger = logging.getLogger(__name__)


class Boundary(object):

    """Boundary objects.
    """

    def __init__(self, lons=None, lats=None, frequency=1):
        self._contour_poly = None
        if lons is not None:
            self.lons = lons[::frequency]
        if lats is not None:
            self.lats = lats[::frequency]

    def contour(self):
        return self.lons, self.lats

    @property
    def contour_poly(self):
        """Get the Spherical polygon corresponding to the Boundary
        """
        if self._contour_poly is None:
            self._contour_poly = SphPolygon(
                np.deg2rad(np.vstack(self.contour()).T))
        return self._contour_poly

    def draw(self, mapper, options, **more_options):
        """Draw the current boundary on the *mapper*
        """
        self.contour_poly.draw(mapper, options, **more_options)


class AreaBoundary(Boundary):

    """Area boundary objects.
    """

    def __init__(self, *sides):
        Boundary.__init__(self)
        self.sides_lons, self.sides_lats = zip(*sides)
        self.sides_lons = list(self.sides_lons)
        self.sides_lats = list(self.sides_lats)

    def decimate(self, ratio):
        """Remove some points in the boundaries, but never the corners.
        """
        for i in range(len(self.sides_lons)):
            length = len(self.sides_lons[i])
            start = int((length % ratio) / 2)
            points = np.concatenate(([0], np.arange(start, length, ratio),
                                     [length - 1]))
            if points[1] == 0:
                points = points[1:]
            if points[-2] == (length - 1):
                points = points[:-1]
            self.sides_lons[i] = self.sides_lons[i][points]
            self.sides_lats[i] = self.sides_lats[i][points]

    def contour(self):
        """Get the (lons, lats) tuple of the boundary object.
        """
        lons = np.concatenate([lns[:-1] for lns in self.sides_lons])
        lats = np.concatenate([lts[:-1] for lts in self.sides_lats])

        return lons, lats


class AreaDefBoundary(AreaBoundary):
    """Boundaries for area definitions (pyresample).
    """

    def __init__(self, area, frequency=1):
        lons, lats = area.get_bbox_lonlats()
        AreaBoundary.__init__(self,
                              *zip(lons, lats))

        if frequency != 1:
            self.decimate(frequency)


class SimpleBoundary(object):
    """Container for geometry boundary.
    Labelling starts in upper left corner and proceeds clockwise"""

    def __init__(self, side1, side2, side3, side4):
        self.side1 = side1
        self.side2 = side2
        self.side3 = side3
        self.side4 = side4
