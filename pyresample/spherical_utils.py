#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2021 Pyresample developers

# Author(s):

#   Adam Dybbroe <Firstname.Lastname@smhi.se>

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

"""
"""

import itertools


def intersects(set1, set2):
    """Does two sets intersects (have anything in common)?"""

    if set1 != set1.difference(set2):
        return True

    return False


# def make_tupled_keys(adict):
#     """Make the dictionary keys tuples of int."""
#     newdict = {}
#     for key, value in adict.items():
#         if isinstance(key, tuple):
#             newdict[key] = value
#         elif isinstance(key, int):
#             newdict[(key,)] = value
#         else:
#             raise KeyError("Dictionary keys must be either tuples of integers or an integer!")

#     return newdict


def find_union_pair(polygons):
    """From a set of polygons find a pair that overlaps.

    *polygons* is a numbered dict with SphPolygon polygons
    If no pair of two polygons overlap return None
    """

    if len(polygons) == 1:
        return None

    for id_, komb_pair in zip(itertools.combinations(polygons.keys(), 2),
                              itertools.combinations(polygons.values(), 2)):
        if intersects(komb_pair[0], komb_pair[1]):
            return id_, komb_pair[0].union(komb_pair[1])

    return None


def merge_unions(polygons):
    """Merge all polygon unions so all polygons left have no intersections with each other."""

    retv = find_union_pair(polygons)
    if retv is None:
        return polygons

    cc_poly = polygons.copy()

    # Go through the dictionary of polygons and merge overlapping ones until
    # there is no change anymore. That is, the input dict of polygons is the
    # same as the output:

    for idx in [0, 1]:
        del cc_poly[retv[0][idx]]
    cc_poly[retv[0]] = retv[1]

    return merge_unions(cc_poly)
