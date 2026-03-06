"""Functions and tools for subsetting a geometry object."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from pyproj import Proj

# this caching module imports the geometries so this subset module
# must be imported inside functions in the geometry modules if needed
# to avoid circular dependencies
from pyresample._caching import cache_to_json_if
from pyresample.boundary import Boundary
from pyresample.geometry import (
    DEFAULT_AREA_SLICE_SAMPLE_STEPS,
    get_geostationary_bounding_box_in_lonlats,
    logger,
)
from pyresample.utils import check_slice_orientation

if TYPE_CHECKING:
    from pyresample import AreaDefinition


MAX_POINTS_PER_CHUNK = 600_000


@cache_to_json_if("cache_geometry_slices")
def get_area_slices(
        src_area: AreaDefinition,
        area_to_cover: AreaDefinition,
        shape_divisible_by: int | None,
        sample_steps: int | None = DEFAULT_AREA_SLICE_SAMPLE_STEPS,
        sample_grid: bool = False,
) -> tuple[slice, slice]:
    """Compute the slice to read based on an `area_to_cover`.

    For geostationary source areas in cross-projection mode:
    - ``sample_steps`` >= 2 and ``sample_grid=False`` samples edge points only.
    - ``sample_steps`` >= 2 and ``sample_grid=True`` samples an interior grid.
    - ``sample_steps`` <= 0 or ``None`` samples all destination points.
    - ``sample_steps`` == 1 raises ``ValueError``.
    """
    if not _is_area_like(src_area):
        raise NotImplementedError(f"Only AreaDefinitions are supported, not {type(src_area)}")
    if not _is_area_like(area_to_cover):
        raise NotImplementedError(f"Only AreaDefinitions are supported, not {type(area_to_cover)}")

    normalized_sample_steps = _normalize_sample_steps(sample_steps)

    # Intersection is only required for two different projections.
    src_crs_wkt = getattr(src_area, "crs_wkt", None)
    dst_crs_wkt = getattr(area_to_cover, "crs_wkt", None)
    if src_crs_wkt is not None and src_crs_wkt == dst_crs_wkt:
        proj_def_to_cover = src_crs_wkt
    else:
        proj_def_to_cover = area_to_cover.crs
        if proj_def_to_cover != src_area.crs:
            proj_def_to_cover = None
    if proj_def_to_cover is not None:
        logger.debug('Projections for data and slice areas are identical: %s',
                     proj_def_to_cover)
        # Get slice parameters
        xstart, xstop, ystart, ystop = _get_slice_starts_stops(src_area, area_to_cover)

        x_slice = check_slice_orientation(slice(xstart, xstop))
        y_slice = check_slice_orientation(slice(ystart, ystop))
        x_slice = _ensure_integer_slice(x_slice)
        y_slice = _ensure_integer_slice(y_slice)
        return x_slice, y_slice

    if src_area.is_geostationary:
        coverage_slices = _get_covered_source_slices(
            src_area,
            area_to_cover,
            sample_steps=normalized_sample_steps,
            sample_grid=sample_grid,
        )
        if coverage_slices is not None:
            return _finalize_slices(src_area, coverage_slices[0], coverage_slices[1], shape_divisible_by)

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

    return _finalize_slices(src_area, x_slice, y_slice, shape_divisible_by)


def _normalize_sample_steps(sample_steps: int | None):
    """Normalize sampling config to sampled mode or dense fallback.

    ``None`` and values ``<= 0`` map to dense destination sampling.
    A value of ``1`` is rejected because it does not provide meaningful sampled
    coverage for edge or grid modes.
    """
    if sample_steps is None:
        return None
    try:
        normalized_sample_steps = int(sample_steps)
    except (TypeError, ValueError) as err:
        raise ValueError(f"sample_steps must be an integer or None, got {sample_steps!r}") from err
    if normalized_sample_steps <= 0:
        return None
    if normalized_sample_steps == 1:
        raise ValueError("sample_steps=1 is not supported; use <= 0/None for dense or >= 2 for sampled modes.")
    return normalized_sample_steps


def _finalize_slices(src_area: AreaDefinition, x_slice: slice, y_slice: slice, shape_divisible_by: int | None):
    """Normalize slice bounds and apply orientation/divisibility rules."""
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


def _get_covered_source_slices(
        src_area: AreaDefinition,
        area_to_cover: AreaDefinition,
        sample_steps: int | None,
        sample_grid: bool,
):
    """Estimate covering source slices from sampled destination points.

    Returns ``None`` when sampled points do not produce any valid source
    coverage, allowing the caller to fall back to boundary intersection.
    """
    min_col = None
    max_col = None
    min_row = None
    max_row = None
    try:
        src_proj = Proj(src_area.crs)
        for destination_lons, destination_lats in _iter_destination_lonlat_samples(
            area_to_cover=area_to_cover,
            sample_steps=sample_steps,
            sample_grid=sample_grid,
        ):
            source_xs, source_ys = src_proj(
                destination_lons,
                destination_lats,
            )
            source_cols, source_rows = src_area.get_array_indices_from_projection_coordinates(
                source_xs,
                source_ys,
            )
            valid = ~np.ma.getmaskarray(source_cols) & ~np.ma.getmaskarray(source_rows)
            if not valid.any():
                continue
            chunk_cols = np.ma.getdata(source_cols)[valid]
            chunk_rows = np.ma.getdata(source_rows)[valid]
            chunk_min_col = int(chunk_cols.min())
            chunk_max_col = int(chunk_cols.max())
            chunk_min_row = int(chunk_rows.min())
            chunk_max_row = int(chunk_rows.max())
            min_col = chunk_min_col if min_col is None else min(min_col, chunk_min_col)
            max_col = chunk_max_col if max_col is None else max(max_col, chunk_max_col)
            min_row = chunk_min_row if min_row is None else min(min_row, chunk_min_row)
            max_row = chunk_max_row if max_row is None else max(max_row, chunk_max_row)
    except (RuntimeError, TypeError, ValueError):
        logger.debug("Failed to estimate covered source slices from sampled destination points.", exc_info=True)
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


def _iter_destination_lonlat_samples(
        area_to_cover: AreaDefinition,
        sample_steps: int | None,
        sample_grid: bool,
):
    """Yield destination lon/lat samples for dense, grid, or edge mode."""
    if sample_steps is None:
        yield from _iter_dense_lonlat_samples(area_to_cover)
        return
    if sample_grid:
        yield _get_grid_lonlat_samples(area_to_cover, sample_steps)
        return
    yield _get_edge_lonlat_samples(area_to_cover, sample_steps)


def _iter_dense_lonlat_samples(area_to_cover: AreaDefinition):
    """Yield full destination lon/lat coverage in row chunks."""
    row_block_size = max(1, MAX_POINTS_PER_CHUNK // area_to_cover.width)
    for row_start in range(0, area_to_cover.height, row_block_size):
        row_stop = min(area_to_cover.height, row_start + row_block_size)
        yield area_to_cover.get_lonlats(
            data_slice=(slice(row_start, row_stop), slice(None)),
            dtype=np.float32,
        )


def _get_grid_lonlat_samples(area_to_cover: AreaDefinition, sample_steps: int):
    """Return one evenly spaced interior destination sample grid."""
    sample_rows = _get_sample_indices(area_to_cover.height, sample_steps)
    sample_cols = _get_sample_indices(area_to_cover.width, sample_steps)
    return area_to_cover.get_lonlats(
        data_slice=(sample_rows[:, None], sample_cols[None, :]),
        dtype=np.float32,
    )


def _get_edge_lonlat_samples(area_to_cover: AreaDefinition, sample_steps: int):
    """Return perimeter destination samples for edge-only sampling mode."""
    if area_to_cover.is_geostationary:
        # Use limb-aware geostationary boundary sampling instead of raw array
        # corners/edges. Projection corners may be off-earth and can under-cover.
        return get_geostationary_bounding_box_in_lonlats(
            area_to_cover,
            nb_points=max(4, sample_steps * 4),
        )
    sample_rows = _get_sample_indices(area_to_cover.height, sample_steps)
    sample_cols = _get_sample_indices(area_to_cover.width, sample_steps)
    top_rows = np.zeros(sample_cols.size, dtype=np.int64)
    top_cols = sample_cols
    right_rows = sample_rows[1:]
    right_cols = np.full(right_rows.size, area_to_cover.width - 1, dtype=np.int64)
    bottom_cols = sample_cols[-2::-1]
    bottom_rows = np.full(bottom_cols.size, area_to_cover.height - 1, dtype=np.int64)
    left_rows = sample_rows[-2:0:-1]
    left_cols = np.zeros(left_rows.size, dtype=np.int64)
    edge_rows = np.concatenate((top_rows, right_rows, bottom_rows, left_rows))
    edge_cols = np.concatenate((top_cols, right_cols, bottom_cols, left_cols))
    return area_to_cover.get_lonlats(data_slice=(edge_rows, edge_cols), dtype=np.float32)


def _get_sample_indices(axis_size: int, sample_steps: int):
    """Return evenly spaced integer sample indices including both endpoints."""
    if sample_steps >= axis_size:
        return np.arange(axis_size, dtype=np.int64)
    return (np.arange(sample_steps, dtype=np.int64) * (axis_size - 1)) // (sample_steps - 1)


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
    """Round slice bounds outward to conservatively preserve target coverage."""
    start = sli.start
    stop = sli.stop
    step = sli.step
    return slice(
        math.floor(start) if start is not None else None,
        math.ceil(stop) if stop is not None else None,
        math.floor(step) if step is not None else None
    )
