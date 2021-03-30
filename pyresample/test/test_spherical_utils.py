#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2021 Adam.Dybbroe

# Author(s):

#   Adam.Dybbroe <a000680@c21856.ad.smhi.se>

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


from pyresample.spherical_utils import find_union_pair
from pyresample.spherical_utils import merge_unions
#from pyresample.spherical_utils import make_tupled_keys

import itertools
import unittest
import numpy as np


SET_A = {1, 3, 5, 7, 9}
SET_B = {2, 4, 6, 8, 10}
SET_C = {12, 14, 16, 18}
SET_D = set(range(10, 20))
SET_E = set(range(20, 30, 3))
SET_F = set(range(21, 30, 3))
SET_G = set(range(22, 30, 2))


class TestPolygonUnions(unittest.TestCase):

    def setUp(self):
        pass

    # def test_make_tupled_keys(self):
    #     """Test changing a dictionary with interger keys into having keys of tupled integers."""

    #     listed_sets = dict(enumerate((SET_A, SET_B)))
    #     retv = make_tupled_keys(listed_sets)

    #     assert retv == {(0,): {1, 3, 5, 7, 9}, (1,): {2, 4, 6, 8, 10}}

    #     mixed_key_sets = {0: {1, 3, 5, 7, 9}, 2: {16, 18, 12, 14},
    #                       (1, 3): {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}}
    #     retv = make_tupled_keys(mixed_key_sets)

    #     assert retv == {(0,): {1, 3, 5, 7, 9}, (2,): {16, 18, 12, 14},
    #                     (1, 3): {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}}

    def test_find_union_pairs(self):
        """Test finding a union pair."""

        listed_sets = dict(enumerate((SET_A, )))
        retv = find_union_pair(listed_sets)

        assert retv is None

        listed_sets = dict(enumerate((SET_A, SET_B)))
        retv = find_union_pair(listed_sets)

        assert retv is None

        listed_sets = dict(enumerate((SET_C, SET_D)))
        retv = find_union_pair(listed_sets)

        assert retv == ((0, 1), set(range(10, 20)))

        listed_sets = dict(enumerate((SET_A, SET_B, SET_D)))
        retv = find_union_pair(listed_sets)

        assert retv == ((1, 2), {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19})

        listed_sets = dict(enumerate((SET_A, SET_B, SET_C, SET_D)))
        retv = find_union_pair(listed_sets)

        assert retv == ((1, 3), {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19})

        listed_sets = {0: {1, 3, 5, 7, 9},
                       4: {0, 10},
                       (2, (1, 3)): {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}}

        retv = find_union_pair(listed_sets)
        assert retv == ((4, (2, (1, 3))), {0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19})

        listed_sets = {0: {1, 3, 5, 7, 9},
                       (2, (1, 3)): {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}}

        retv = find_union_pair(listed_sets)

        assert retv is None

    # def test_find_union_pairs_tuplekeys(self):
    #     """Test finding union pairs if one key is a tuple."""

    #     mixed_key_sets = {0: {1, 3, 5, 7, 9}, 2: {16, 18, 12, 14},
    #                       (1, 3): {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}}
    #     retv = find_union_pair(mixed_key_sets)

    #     # assert retv == ((2, 1, 3), {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19})
    #     assert retv == ((2, (1, 3)), {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19})

    #     mixed_key_sets = {0: {1, 3, 5, 7, 9},
    #                       (2, (1, 3)): {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}}

    #     retv = find_union_pair(mixed_key_sets)

    #     assert retv is None

    def test_merge_unions(self):
        """Test merging union pairs iteratively"""

        listed_sets = dict(enumerate((SET_A, SET_B, SET_C)))
        retv = merge_unions(listed_sets)

        assert retv == {0: SET_A, 1: SET_B, 2: SET_C}

        listed_sets = dict(enumerate((SET_C, SET_D)))
        retv = merge_unions(listed_sets)

        assert retv == {(0, 1): set(range(10, 20))}

        listed_sets = dict(enumerate((SET_A, SET_B, SET_C, SET_D)))
        retv = merge_unions(listed_sets)

        assert retv == {0: {1, 3, 5, 7, 9},
                        (2, (1, 3)): {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}}

        listed_sets = {0: {1, 3, 5, 7, 9},
                       (2, (1, 3)): {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}}
        retv = merge_unions(listed_sets)

        assert retv == listed_sets

        listed_sets = dict(enumerate((SET_A, SET_B, SET_C, SET_D, SET_E, SET_F, SET_G)))
        retv = merge_unions(listed_sets)

        assert retv == {0: {1, 3, 5, 7, 9},
                        (2, (1, 3)): {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
                        (5, (4, 6)): {20, 21, 22, 23, 24, 26, 27, 28, 29}}
