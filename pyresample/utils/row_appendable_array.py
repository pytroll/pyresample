#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010-2020 Pyresample developers
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Appendable numpy array with allows for efficient pre-allocation."""

import numpy as np


class RowAppendableArray:
    """Helper class which allows efficient concatenation of numpy arrays bey pre-allocating buffers.

    By default, this class behaves the same as subsequent array concatenations.
    """

    def __init__(self, expected_segments):
        """Create an appendable array by lazily pre-allocating the array buffer.

        The size of the buffer depends on the expected number of segments and the size of the first segment.
        """
        self._expected_segments = expected_segments
        self._data = None
        self._cursor = 0

    def append_row(self, next_array):
        """Append the specified array."""
        if self._data is None:
            self._data = np.empty((self._expected_segments * next_array.shape[0], *next_array.shape[1:]),
                                  dtype=next_array.dtype)
        cursor_end = self._cursor + next_array.shape[0]
        if cursor_end > self._data.shape[0]:
            if len(next_array.shape) == 1:
                self._data = np.append(self._data, next_array)
            else:
                self._data = np.row_stack((self._data, next_array))
        else:
            self._data[self._cursor:cursor_end] = next_array
        self._cursor = cursor_end

    def to_array(self):
        """Return the numpy array with all the data appended until now."""
        return self._data[:self._cursor]
