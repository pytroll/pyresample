"""Functions and tools for subsetting a geometry object."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

# this caching module imports the geometries so this subset module
# must be imported inside functions in the geometry modules if needed
# to avoid circular dependencies
from pyresample._caching import cache_to_json_if
from pyresample.boundary import Boundary
from pyresample.geometry import get_geostationary_bounding_box_in_lonlats, logger
from pyresample.utils import check_slice_orientation

if TYPE_CHECKING:
    from pyresample import AreaDefinition


@cache_to_json_if("cache_geometry_slices")
def get_area_slices(
        src_area: AreaDefinition,
        area_to_cover: AreaDefinition,
        shape_divisible_by: int | None,
) -> tuple[slice, slice]:
    """Compute the slice to read based on an `area_to_cover`."""
    if not _is_area_like(src_area):
        raise NotImplementedError(f"Only AreaDefinitions are supported, not {type(src_area)}")
    if not _is_area_like(area_to_cover):
        raise NotImplementedError(f"Only AreaDefinitions are supported, not {type(area_to_cover)}")

    # Intersection only required for two different projections
    proj_def_to_cover = area_to_cover.crs
    proj_def = src_area.crs
    if proj_def_to_cover == proj_def:
        logger.debug('Projections for data and slice areas are identical: %s',
                     proj_def_to_cover)
        # Get slice parameters
        xstart, xstop, ystart, ystop = _get_slice_starts_stops(src_area, area_to_cover)

        x_slice = check_slice_orientation(slice(xstart, xstop))
        y_slice = check_slice_orientation(slice(ystart, ystop))
        x_slice = _ensure_integer_slice(x_slice)
        y_slice = _ensure_integer_slice(y_slice)
        return x_slice, y_slice

    data_boundary = _get_area_boundary(src_area)
    area_boundary = _get_area_boundary(area_to_cover)
    intersection = data_boundary.contour_poly.intersection(
        area_boundary.contour_poly)
    if intersection is None:
        logger.debug('Cannot determine appropriate slicing. '
                     "Data and projection area do not overlap.")
        raise NotImplementedError
    x, y = src_area.get_array_indices_from_lonlat(
        np.rad2deg(intersection.lon), np.rad2deg(intersection.lat))
    x_slice = slice(np.ma.min(x), np.ma.max(x) + 1)
    y_slice = slice(np.ma.min(y), np.ma.max(y) + 1)
    if src_area.is_geostationary:
        coverage_slices = _get_covered_source_slices(src_area, area_to_cover)
        if coverage_slices is not None:
            x_slice = _merge_slices(x_slice, coverage_slices[0])
            y_slice = _merge_slices(y_slice, coverage_slices[1])
    x_slice = _ensure_integer_slice(x_slice)
    y_slice = _ensure_integer_slice(y_slice)
    if shape_divisible_by is not None:
        x_slice = _make_slice_divisible(x_slice, src_area.width,
                                        factor=shape_divisible_by)
        y_slice = _make_slice_divisible(y_slice, src_area.height,
                                        factor=shape_divisible_by)

    return (check_slice_orientation(x_slice),
            check_slice_orientation(y_slice))


def _is_area_like(area_obj: Any) -> bool:
    return hasattr(area_obj, "crs") and hasattr(area_obj, "area_extent")


def _get_slice_starts_stops(src_area, area_to_cover):
    """Get x and y start and stop points for slicing."""
    llx, lly, urx, ury = area_to_cover.area_extent
    x, y = src_area.get_array_coordinates_from_projection_coordinates([llx, urx], [lly, ury])

    # we use `round` because we want the *exterior* of the pixels to contain the area_to_cover's area extent.
    if (src_area.area_extent[0] > src_area.area_extent[2]) ^ (llx > urx):
        xstart = max(0, round(x[1]))
        xstop = min(src_area.width, round(x[0]) + 1)
    else:
        xstart = max(0, round(x[0]))
        xstop = min(src_area.width, round(x[1]) + 1)
    if (src_area.area_extent[1] > src_area.area_extent[3]) ^ (lly > ury):
        ystart = max(0, round(y[0]))
        ystop = min(src_area.height, round(y[1]) + 1)
    else:
        ystart = max(0, round(y[1]))
        ystop = min(src_area.height, round(y[0]) + 1)

    return xstart, xstop, ystart, ystop


def _get_area_boundary(area_to_cover: AreaDefinition) -> Boundary:
    try:
        if area_to_cover.is_geostationary:
            return Boundary(*get_geostationary_bounding_box_in_lonlats(area_to_cover))
        boundary_shape = max(max(*area_to_cover.shape) // 100 + 1, 3)
        return area_to_cover.boundary(frequency=boundary_shape, force_clockwise=True)
    except ValueError as err:
        raise NotImplementedError("Can't determine boundary of area to cover") from err


def _get_covered_source_slices(src_area: AreaDefinition, area_to_cover: AreaDefinition) -> tuple[slice, slice] | None:
    max_points_per_chunk = 600_000
    row_block_size = max(1, max_points_per_chunk // area_to_cover.width)
    min_col = None
    max_col = None
    min_row = None
    max_row = None
    try:
        for row_start in range(0, area_to_cover.height, row_block_size):
            row_stop = min(area_to_cover.height, row_start + row_block_size)
            destination_lons, destination_lats = area_to_cover.get_lonlats(
                data_slice=(slice(row_start, row_stop), slice(None)),
                dtype=np.float32,
            )
            source_cols, source_rows = src_area.get_array_indices_from_lonlat(
                destination_lons,
                destination_lats,
            )
            valid = ~np.ma.getmaskarray(source_cols) & ~np.ma.getmaskarray(source_rows)
            if not np.any(valid):
                continue
            source_cols = np.ma.array(source_cols, mask=~valid, copy=False)
            source_rows = np.ma.array(source_rows, mask=~valid, copy=False)
            chunk_min_col = int(np.ma.min(source_cols))
            chunk_max_col = int(np.ma.max(source_cols))
            chunk_min_row = int(np.ma.min(source_rows))
            chunk_max_row = int(np.ma.max(source_rows))
            min_col = chunk_min_col if min_col is None else min(min_col, chunk_min_col)
            max_col = chunk_max_col if max_col is None else max(max_col, chunk_max_col)
            min_row = chunk_min_row if min_row is None else min(min_row, chunk_min_row)
            max_row = chunk_max_row if max_row is None else max(max_row, chunk_max_row)
    except (RuntimeError, TypeError, ValueError):
        return None
    if min_col is None or min_row is None or max_col is None or max_row is None:
        return None
    col_start = max(0, min_col)
    col_stop = min(src_area.width, max_col + 1)
    row_start = max(0, min_row)
    row_stop = min(src_area.height, max_row + 1)
    if col_start >= col_stop or row_start >= row_stop:
        return None
    return slice(col_start, col_stop), slice(row_start, row_stop)


def _merge_slices(base_slice: slice, other_slice: slice) -> slice:
    return slice(
        min(base_slice.start, other_slice.start),
        max(base_slice.stop, other_slice.stop),
    )


def _make_slice_divisible(sli, max_size, factor=2):
    """Make the given slice even in size."""
    rem = (sli.stop - sli.start) % factor
    if rem != 0:
        adj = factor - rem
        if sli.stop + 1 + rem < max_size:
            sli = slice(sli.start, sli.stop + adj)
        elif sli.start > 0:
            sli = slice(sli.start - adj, sli.stop)
        else:
            sli = slice(sli.start, sli.stop - rem)

    return sli


def _ensure_integer_slice(sli):
    start = sli.start
    stop = sli.stop
    step = sli.step
    return slice(
        math.floor(start) if start is not None else None,
        math.ceil(stop) if stop is not None else None,
        math.floor(step) if step is not None else None
    )
