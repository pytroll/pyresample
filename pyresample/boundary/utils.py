#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2014-2023 Pyresample developers
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
"""Utility to extract boundary mask and indices."""
import numpy as np 


def _find_boundary_mask(mask): 
    """Find boundary of a binary mask."""
    mask = mask.astype(int)
    # Pad with zeros (enable to detect Trues at mask boundaries)
    padded_mask = np.pad(mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    # Shift the image in four directions and compare with the original
    shift_up = np.roll(padded_mask, -1, axis=0)
    shift_down = np.roll(padded_mask, 1, axis=0)
    shift_left = np.roll(padded_mask, -1, axis=1)
    shift_right = np.roll(padded_mask, 1, axis=1)

    # Find the boundary points
    padded_boundary_mask = ((padded_mask != shift_up) | (padded_mask != shift_down) | 
                   (padded_mask != shift_left) | (padded_mask != shift_right)) & padded_mask

    boundary_mask = padded_boundary_mask[1:-1,1:-1]
    return boundary_mask


def find_boundary_mask(lons, lats):
    """Find the boundary mask."""
    valid_mask = np.isfinite(lons) & np.isfinite(lats)
    return _find_boundary_mask(valid_mask)
     

def get_ordered_contour(contour_mask):
    """Return the ordered indices of a contour mask."""
    # Count number of rows and columns 
    rows, cols = contour_mask.shape
    # Function to find the next contour point
    def next_point(current, last, visited):
        for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1), (1, -1), (-1, -1), (1, 1), (-1, 1)]:
            next_pt = (current[0] + dx, current[1] + dy)
            if next_pt != last and next_pt not in visited and 0 <= next_pt[0] < rows and 0 <= next_pt[1] < cols and contour_mask[next_pt]:
                return next_pt
        return None
    # Initialize
    contour = []
    visited = set()  # Keep track of visited points
    # Find the starting point
    start = tuple(np.argwhere(contour_mask)[0])
    contour.append(start)
    visited.add(start)
    # Initialize last and current points
    last = start
    current = next_point(last, None, visited)
    while current and current != start:
        contour.append(current)
        visited.add(current)
        next_pt = next_point(current, last, visited)
        if not next_pt:  # Break if no next point found
            break
        last, current = current, next_pt
    return np.array(contour)


def find_boundary_contour_indices(lons, lats):
    """Find the boundary contour ordered indices."""
    boundary_mask = find_boundary_mask(lons, lats)
    boundary_contour_idx = get_ordered_contour(boundary_mask)
    return boundary_contour_idx
    