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
"""Test for spherical calculations."""

from unittest.mock import patch

import numpy as np
import pytest

from pyresample.spherical import SphPolygon
from pyresample.spherical_utils import (
    GetNonOverlapUnionsBaseClass,
    check_if_two_polygons_overlap,
    check_keys_int_or_tuple,
    merge_tuples,
)

SET_A = {1, 3, 5, 7, 9}
SET_B = {2, 4, 6, 8, 10}
SET_C = {12, 14, 16, 18}
SET_D = set(range(10, 20))
SET_E = set(range(20, 30, 3))
SET_F = set(range(21, 30, 3))
SET_G = set(range(22, 30, 2))


def fake_merge_tuples(intuple):
    """Fake the merge tuples method."""
    if intuple == (2, (1, 3)):
        return (2, 1, 3)
    if intuple == (5, (4, 6)):
        return (5, 4, 6)

    return None


def test_check_overlap_one_polygon_entirely_inide_another():
    """Test the function to check if two polygons overlap each other.

    In this case one polygon is entirely inside the other.
    """
    vertices = np.array([[1, 1, 20, 20],
                         [1, 20, 20, 1]]).T
    poly1 = SphPolygon(np.deg2rad(vertices))
    vertices = np.array([[0, 0, 30, 30],
                         [0, 30, 30, 0]]).T
    poly2 = SphPolygon(np.deg2rad(vertices))

    result = check_if_two_polygons_overlap(poly1, poly2)
    assert result is True


def test_check_overlap_one_polygon_not_entirely_inside_another():
    """Test the function to check if two polygons overlap each other.

    In this case one polygon is not entirely inside the other but they overlap
    each other.
    """
    vertices = np.array([[180, 90, 0, -90],
                         [89, 89, 89, 89]]).T
    poly1 = SphPolygon(np.deg2rad(vertices))

    vertices = np.array([[-45, -135, 135, 45],
                         [89, 89, 89, 89]]).T
    poly2 = SphPolygon(np.deg2rad(vertices))

    res = check_if_two_polygons_overlap(poly1, poly2)
    assert res is True


def test_check_overlap_two_polygons_having_no_overlap():
    """Test the function to check if two polygons overlap each other.

    In this case the two polygons do not have any overlap.
    """
    vertices = np.array([[10, 10, 20, 20],
                         [10, 20, 20, 10]]).T
    poly1 = SphPolygon(np.deg2rad(vertices))
    vertices = np.array([[25, 25, 40, 40],
                         [25, 40, 40, 25]]).T
    poly2 = SphPolygon(np.deg2rad(vertices))

    res = check_if_two_polygons_overlap(poly1, poly2)

    assert res is False


@patch('pyresample.spherical_utils.check_keys_int_or_tuple')
def test_merge_when_input_objects_do_not_overlap(keys_int_or_tuple):
    """Test main method (merge) of the GetNonOverlapUnionsBaseClass class."""
    mysets = [{1, 3, 5, 7, 9}, {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}]
    myobjects = GetNonOverlapUnionsBaseClass(mysets)

    keys_int_or_tuple.return_code = None

    with patch('pyresample.spherical_utils.merge_tuples', return_value=0):
        myobjects.merge()

    polygons = myobjects.get_polygons()

    assert polygons == mysets


@patch('pyresample.spherical_utils.check_keys_int_or_tuple')
def test_merge_overlapping_and_nonoverlapping_objects(keys_int_or_tuple):
    """Test main method (merge) of the GetNonOverlapUnionsBaseClass class."""
    mysets = [SET_A, SET_B, SET_C, SET_D, SET_E, SET_F, SET_G]
    myobjects = GetNonOverlapUnionsBaseClass(mysets)

    keys_int_or_tuple.return_code = None

    with patch('pyresample.spherical_utils.merge_tuples') as mypatch:
        mypatch.side_effect = fake_merge_tuples
        myobjects.merge()

    polygons = myobjects.get_polygons()
    ids = myobjects.get_ids()

    polygons_expected = [{1, 3, 5, 7, 9},
                         {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
                         {20, 21, 22, 23, 24, 26, 27, 28, 29}]

    assert polygons == polygons_expected

    ids_expected = [0, (2, 1, 3), (5, 4, 6)]

    assert ids == ids_expected


def test_flatten_tuple_input_requires_tuple():
    """Test flatten a nested tuple of integers.

    Input a scalar (not a tuple).

    """
    with pytest.raises(TypeError) as exec_info:
        _ = merge_tuples(7)

    exception_raised = exec_info.value
    assert str(exception_raised) == "Function argument must be a tuple!"


def test_flatten_tuple_input_1tuple():
    """Test flatten a nested tuple of integers.

    Input a tuple of one scalar.

    """
    intuple = (0,)
    res = merge_tuples(intuple)
    assert res == (0,)


def test_flatten_tuple_input_2tuple_of_2tuples_of_2tuples():
    """Test flatten a nested tuple of integers.

    Input a 2-tuple of 2-tuples of 2-tuples

    """
    intuple = (((0, 1), (2, 3)), ((4, 5), (6, 7)))
    res = merge_tuples(intuple)
    assert res == (0, 1, 2, 3, 4, 5, 6, 7)


def test_flatten_tuple_input_2tuple_of_scalar_and_2tuple():
    """Test flatten a nested tuple of integers.

    Input a 2-tuple of a scalar and a 2-tuple

    """
    intuple = (3, (1, 2))
    res = merge_tuples(intuple)
    assert res == (3, 1, 2)


def test_flatten_tuple_input_3tuple_of_scalar_and_2tuple_and_2tuple_of_scalar_and_tuple():
    """Test flatten a nested tuple of integers.

    Input a 3-tuple of a scalar, a 2-tuple and a 2-tuple of a scalar and a tuple

    """
    intuple = (0, (1, 2), (3, (4, 5)))
    res = merge_tuples(intuple)
    assert res == (0, 1, 2, 3, 4, 5)


def test_flatten_tuple_input_tuple_of_scalar_and_2tuple_and_2tuple_of_scalar_and_tuple_and_1tuple():
    """Test flatten a nested tuple of integers.

    Input a tuple of a scalar, a 2-tuple, a 2-tuple of a scalar and a tuple, and a 1-tuple.

    """
    intuple = (0, (1, 2), (3, (4, 5)), (6,))
    res = merge_tuples(intuple)
    assert res == (0, 1, 2, 3, 4, 5, 6)


def test_find_union_pairs_input_one_set():
    """Test finding a union pair.

    In this case input only one listed set (~polygon).

    """
    listed_sets = dict(enumerate((SET_A, )))
    this = GetNonOverlapUnionsBaseClass(listed_sets)
    retv = this._find_union_pair(listed_sets)

    assert retv is None


def test_find_union_pairs_input_two_non_overlapping_sets():
    """Test finding a union pair.

    In this case input two non-overlapping sets giving no union.

    """
    listed_sets = dict(enumerate((SET_A, SET_B)))
    this = GetNonOverlapUnionsBaseClass(listed_sets)
    retv = this._find_union_pair(listed_sets)

    assert retv is None


def test_find_union_pairs_input_two_overlapping_sets():
    """Test finding a union pair.

    In this case input two overlapping sets.

    """
    listed_sets = dict(enumerate((SET_C, SET_D)))
    this = GetNonOverlapUnionsBaseClass(listed_sets)
    retv = this._find_union_pair(listed_sets)

    assert retv == ((0, 1), set(range(10, 20)))


def test_find_union_pairs_input_three_sets_one_entirely_without_overlap():
    """Test finding a union pair.

    In this case input three sets where one does not overlap the two others
    which in turn do overlap each other.

    """
    listed_sets = dict(enumerate((SET_A, SET_B, SET_D)))
    this = GetNonOverlapUnionsBaseClass(listed_sets)
    retv = this._find_union_pair(listed_sets)

    assert retv == ((1, 2), {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19})


def test_find_union_pairs_input_four_sets_where_only_two_have_overlap():
    """Test finding a union pair.

    In this case input four sets where two overlap and the other two does not
    overlap with any other.

    """
    listed_sets = dict(enumerate((SET_A, SET_B, SET_C, SET_D)))
    this = GetNonOverlapUnionsBaseClass(listed_sets)
    retv = this._find_union_pair(listed_sets)

    assert retv == ((1, 3), {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19})


def test_find_union_pairs_input_three_sets_one_entirely_without_overlap_one_already_a_union():
    """Test finding a union pair.

    In this case input three sets, one with no overlap of the others, and one
    of the overlapping ones is already a paired union.

    """
    listed_sets = {0: {1, 3, 5, 7, 9},
                   4: {0, 10},
                   (2, (1, 3)): {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}}

    this = GetNonOverlapUnionsBaseClass(listed_sets)
    retv = this._find_union_pair(listed_sets)

    assert retv == ((4, (2, (1, 3))), {0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19})


def test_find_union_pairs_input_two_sets_without_overlap_one_already_a_union():
    """Test finding a union pair.

    In this case input two sets with no overlap, but one is already a paired
    union.

    """
    listed_sets = {0: {1, 3, 5, 7, 9},
                   (2, (1, 3)): {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}}

    this = GetNonOverlapUnionsBaseClass([])
    retv = this._find_union_pair(listed_sets)

    assert retv is None


def test_merge_unions_input_three_sets_without_overlap():
    """Test merging union pairs iteratively.

    In this case input three sets without any overlap.

    """
    listed_sets = dict(enumerate((SET_A, SET_B, SET_C)))
    this = GetNonOverlapUnionsBaseClass(listed_sets)
    retv = this._merge_unions(listed_sets)

    assert retv == {0: SET_A, 1: SET_B, 2: SET_C}


def test_merge_unions_input_two_overlapping_sets():
    """Test merging union pairs iteratively.

    In this case input two overlapping sets.
    """
    listed_sets = dict(enumerate((SET_C, SET_D)))
    this = GetNonOverlapUnionsBaseClass(listed_sets)
    retv = this._merge_unions(listed_sets)

    assert retv == {(0, 1): set(range(10, 20))}


def test_merge_unions_input_four_sets_one_overlapping_two_others():
    """Test merging union pairs iteratively.

    In this case input 4 sets, where one is overlapping two others.

    """
    listed_sets = dict(enumerate((SET_A, SET_B, SET_C, SET_D)))
    this = GetNonOverlapUnionsBaseClass(listed_sets)
    retv = this._merge_unions(listed_sets)

    assert retv == {0: {1, 3, 5, 7, 9},
                    (2, (1, 3)): {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}}


def test_merge_unions_input_two_non_overlapping_sets():
    """Test merging union pairs iteratively.

    In this case input 2 sets that do not overlap, but one is already a paired union.

    """
    listed_sets = {0: {1, 3, 5, 7, 9},
                   (2, (1, 3)): {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}}
    this = GetNonOverlapUnionsBaseClass(listed_sets)
    retv = this._merge_unions(listed_sets)

    assert retv == listed_sets


def test_merge_unions_input_seven_sets_with_overlaps():
    """Test merging union pairs iteratively.

    In this case input 7 sets, several of which overlap each other but one is
    completely unique and has no overlap with any of the other 6.

    """
    listed_sets = dict(enumerate((SET_A, SET_B, SET_C, SET_D, SET_E, SET_F, SET_G)))
    this = GetNonOverlapUnionsBaseClass(listed_sets)
    retv = this._merge_unions(listed_sets)

    assert retv == {0: {1, 3, 5, 7, 9},
                    (2, (1, 3)): {2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
                    (5, (4, 6)): {20, 21, 22, 23, 24, 26, 27, 28, 29}}


def test_check_keys_int_or_tuple_input_okay():
    """Test the check for dictionary keys and input only a dict with the accepted keys of integers and tuples."""
    adict = {1: [1, 2, 3], (2, 3): [1, 2, 3, 4], (6, (4, 5)): [1, 2, 3, 4, 5]}
    check_keys_int_or_tuple(adict)


def test_check_keys_int_or_tuple_input_string():
    """Test the check for dictionary keys and input a dict with a key which is a string."""
    adict = {1: [1, 2, 3], 'set B': [1, 2, 3, 4]}
    with pytest.raises(KeyError) as exec_info:
        check_keys_int_or_tuple(adict)

    exception_raised = exec_info.value

    assert str(exception_raised) == "'Key must be integer or a tuple (of integers)'"


def test_check_keys_int_or_tuple_input_float():
    """Test the check for dictionary keys and input a dict with a key which is a float."""
    adict = {1.1: [1, 2, 3]}
    with pytest.raises(KeyError) as exec_info:
        check_keys_int_or_tuple(adict)

    exception_raised = exec_info.value
    assert str(exception_raised) == "'Key must be integer or a tuple (of integers)'"
