#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Pyresample developers
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
"""Functions to support the calculation of a coverage of an area by a set of spherical polygons.

It can for instance be a set of satellite overpasses to be received of a given
local stations over a certain time window where we want to calculate how much
of an area is covered by the onboard scanning instrument(s).
"""

import itertools


class GetNonOverlapUnionsBaseClass():
    """Base class to get the smallest set of union objects that does not overlap.

    The objects are here Python sets of integers - but are abstracts for
    geometrical shapes on a sphere.

    """

    def __init__(self, geom_objects):
        self.geometries = geom_objects
        self.geom_numbers = range(len(geom_objects))

        self._geoms = dict(enumerate(geom_objects))

    def merge(self):
        """Merge all overlapping objects (sets or polygons)."""
        all_unions = self._merge_unions(self._geoms)
        check_keys_int_or_tuple(all_unions)

        self._geoms = {}
        for key, value in all_unions.items():
            if isinstance(key, int):
                self._geoms[key] = value
            else:
                newkey = merge_tuples(key)
                self._geoms[newkey] = value

    def get_polygons(self):
        """Get a list of all non-overlapping polygon unions."""
        return list(self._geoms.values())

    def get_ids(self):
        """Get a list of identifiers identifying the gemoetry objects in each polygon union."""
        return list(self._geoms.keys())

    def _overlaps(self, set1, set2):
        """Check if the two sets overlap each other (have anything in common)."""
        if set1 != set1.difference(set2):
            return True

        return False

    def _find_union_pair(self, geoms):
        """From a set of geometries find a pair that overlaps.

        *geoms* is here expected to be a numbered dict with SphPolygon
        polygons. If no pair of two polygons overlap (that is the union
        returns something different from None or False) then return None.

        Strictly the geometries/objects does not need to be a SphPolygon. The
        only requirement is that it has a union method with the same behaviour.

        """
        if len(geoms) == 1:
            return None

        for id_, komb_pair in zip(itertools.combinations(geoms.keys(), 2),
                                  itertools.combinations(geoms.values(), 2)):
            if self._overlaps(komb_pair[0], komb_pair[1]):
                return id_, komb_pair[0].union(komb_pair[1])

        return None

    def _merge_unions(self, geoms):
        """Merge all overlapping geometry unions.

        Go through the dictionary of geometries and merge overlapping ones until
        there is no change anymore. That is, the input dict of geometries is the
        same as the output:

        """
        retv = self._find_union_pair(geoms)
        if retv is None:
            return geoms

        cc_poly = geoms.copy()

        for idx in [0, 1]:
            del cc_poly[retv[0][idx]]
        cc_poly[retv[0]] = retv[1]

        return self._merge_unions(cc_poly)


class GetNonOverlapUnions(GetNonOverlapUnionsBaseClass):
    """NonOverlapUnions class."""

    def __init__(self, polygons):
        """Init the GetNonOverlapUnions."""
        super(GetNonOverlapUnions, self).__init__(polygons)

    def _overlaps(self, polygon1, polygon2):
        """Check if two polygons overlap each other (have anything in common).

        Return True if they do overlap, otherwise False.

        *polygon1* and *polygon2* are here expected to be SphPolygon
        polygons. Strictly the two input objects do not need to be of type
        SphPolygon. The only requirement is that they have a union method with
        the same behaviour.

        """
        return check_if_two_polygons_overlap(polygon1, polygon2)


def merge_tuples(atuple):
    """Take a nested tuple of integers and concatenate it to a tuple of integers."""
    if not isinstance(atuple, tuple):
        raise TypeError("Function argument must be a tuple!")

    while True:
        # Test if all items in tuple are scalars and can be summed:
        try:
            _ = sum(atuple)
        except TypeError:
            pass
        else:
            return atuple

        atuple = _int_items_to_tuples(atuple)

        try:
            atuple = sum(atuple, ())
        except TypeError:
            return atuple


def _int_items_to_tuples(mytuple):
    """Turn integer scalars in a tuple into tuples."""
    newtup = []
    for item in mytuple:
        if isinstance(item, int):
            newtup.append((item,))
        else:
            newtup.append(item)

    return tuple(newtup)


def check_keys_int_or_tuple(adict):
    """Check if the dictionary keys are integers or tuples.

    If they are not, raise a KeyError
    """
    for key in adict:
        if not isinstance(key, (int, tuple)):
            raise KeyError("Key must be integer or a tuple (of integers)")


def check_if_two_polygons_overlap(polygon1, polygon2):
    """Check if two SphPolygons overlaps."""
    if polygon1.union(polygon2):
        return True

    return False
