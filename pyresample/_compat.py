#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Pyresample developers
#
# This file is part of pyresample.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Backports and compatibility fixes for pyresample."""

import sys

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points  # noqa
else:
    from importlib.metadata import entry_points  # noqa

try:
    from numpy.typing import ArrayLike  # noqa
except ImportError:
    # numpy <1.20
    from numpy import ndarray as ArrayLike  # noqa
