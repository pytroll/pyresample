# Copyright (C) 2023 Pyresample developers
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
"""Functions for html representation of area definition."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from functools import lru_cache
from html import escape
from importlib.resources import read_binary
from typing import Literal, Optional, Union

import numpy as np

import pyresample.geometry as geom
from pyresample.utils.proj4 import ignore_pyproj_proj_warnings

try:
    import cartopy
except ModuleNotFoundError:
    cartopy = None

try:
    import xarray as xr
    from xarray.core.formatting_html import _obj_repr, datavar_section
    xarray = True
except ModuleNotFoundError:
    xarray = False


STATIC_FILES = (
    ("pyresample.static.html", "icons_svg_inline.html"),
    ("pyresample.static.css", "style.css"),
)


@lru_cache(None)
def _load_static_files():
    """Lazily load the resource files into memory the first time they are needed."""
    return [
        read_binary(package, resource).decode("utf-8")
        for package, resource in STATIC_FILES
    ]


def _icon(icon_name):
    # icon_name should be defined in pyresample/static/html/icon-svg-inline.html
    return (
        "<svg class='icon pyresample-{0}'>"
        "<use xlink:href='#{0}'>"
        "</use>"
        "</svg>".format(icon_name)
    )


def plot_area_def(area: Union['geom.AreaDefinition', 'geom.SwathDefinition'], # noqa F821
                  fmt: Optional[Literal["svg", "png", None]] = None,
                  features: Optional[Iterable[str]] = None,
                  ) -> Union[str, None]:
    """Plot area.

    Args:
        area : Area/Swath to plot.
        fmt : Output format of the plot. The output is the string representation of
            the respective format xml for svg and base64 for png. Either svg or png.
            If None (default) plot is just shown.
        features: Series of string names of cartopy features to add to the plot.
            Can be lowercase or uppercase names of the features, for example,
            "land", "coastline", "borders", "ocean", or any other feature
            available from ``cartopy.feature``. If None (default), then land,
            coastline, and borders are used.

    Returns:
        svg or png image as string or ``None`` when no format is provided
        in which case the plot is shown interactively.

    """
    import base64
    from io import BytesIO, StringIO

    import matplotlib.pyplot as plt

    if isinstance(area, geom.AreaDefinition):
        crs = area.to_cartopy_crs()
        fig, ax = plt.subplots(subplot_kw=dict(projection=crs))
    elif isinstance(area, geom.SwathDefinition):
        from shapely.geometry.polygon import Polygon

        lx, ly = area.get_edge_lonlats()

        crs = cartopy.crs.Mercator()
        fig, ax = plt.subplots(subplot_kw=dict(projection=crs))

        poly = Polygon(list(zip(lx[::-1], ly[::-1])))  # make lat/lon counterclockwise for shapely
        ax.add_geometries([poly], crs=cartopy.crs.CRS(area.crs), facecolor="none", edgecolor="red")
        bounds = poly.buffer(5).bounds
        ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=cartopy.crs.CRS(area.crs))
    else:
        raise NotImplementedError("Only AreaDefinition and SwathDefinition objects can be plotted")

    if features is None:
        features = ("land", "coastline", "borders")

    for feat_name in features:
        feat_obj = getattr(cartopy.feature, feat_name.upper())
        ax.add_feature(feat_obj)

    plt.tight_layout(pad=0)

    if fmt == "svg":
        svg_str = StringIO()
        plt.savefig(svg_str, format="svg", bbox_inches="tight")
        plt.close()
        return svg_str.getvalue()
    elif fmt == "png":
        png_str = BytesIO()
        plt.savefig(png_str, format="png", bbox_inches="tight")
        img_str = f"<img src='data:image/png;base64, {base64.encodebytes(png_str.getvalue()).decode('utf-8')}'/>"
        plt.close()
        return img_str
    else:
        plt.show()
        return None


def collapsible_section(name: str, inline_details: Optional[str] = "", details: Optional[str] = "",
                        enabled: Optional[bool] = True, collapsed: Optional[bool] = False,
                        icon: Optional[str] = None) -> str:
    """Create a collapsible section.

    Args:
      name : Name of the section
      inline_details : Information to show when section is collapsed. Default nothing.
      details : Details to show when section is expanded.
      enabled : Is collapsing enabled. Default True.
      collapsed:  Is the section collapsed on first show. Default False.
      icon : Icon to use for collapsible section.

    Returns:
      Html div structure for collapsible section.

    """
    # "unique" id to expand/collapse the section
    data_id = "section-" + str(uuid.uuid4())

    collapse_enabled = "" if enabled else "disabled"
    is_collapsed = "" if collapsed else "checked"
    tip = " title='Expand/collapse section'" if enabled else ""

    if icon is None:
        icon = _icon("icon-database")

    return ("<div class='pyresample-area-section-item'>"
            f"<input id='{data_id}' class='pyresample-area-section-in' "
            f"type='checkbox' {collapse_enabled} {is_collapsed}>"
            f"<label for='{data_id}' {tip}>{icon} {name}</label>"
            f"<div class='pyresample-area-section-preview'>{inline_details}</div>"
            f"<div class='pyresample-area-section-details'>{details}</div>"
            "</div>"
            )


def proj_area_attrs_section(area: 'geom.AreaDefinition') -> str: # noqa F821
    """Create html for attribute section based on an area Area.

    Args:
        area : Area definition.

    Returns:
        Html with collapsible section of attributes of Area.

    """
    resolution_str = "/".join([str(round(x, 1)) for x in area.resolution])
    with ignore_pyproj_proj_warnings():
        proj_dict = area.proj_dict
    proj_str = "{{{}}}".format(", ".join(["'%s': '%s'" % (str(k), str(proj_dict[k])) for k in
                                          sorted(proj_dict.keys())]))
    area_units = proj_dict.get("units", "")

    attrs_icon = _icon("icon-file-text2")

    area_attrs = ("<dl>"
                  f"<dt>Area name</dt><dd>{area.area_id}</dd>"
                  f"<dt>Description</dt><dd>{area.description}</dd>"
                  f"<dt>Projection</dt><dd>{proj_str}</dd>"
                  f"<dt>Width/Height</dt><dd>{area.width}/{area.height} Pixel</dd>"
                  f"<dt>Resolution x/y (SSP)</dt><dd>{resolution_str} {area_units}</dd>"
                  f"<dt>Extent (ll_x, ll_y, ur_x, ur_y)</dt>"
                  f"<dd>{tuple(round(float(x), 4) for x in area.area_extent)}</dd>"
                  "</dl>"
                  )

    coll = collapsible_section("Properties", details=area_attrs, icon=attrs_icon)

    return f"{coll}"


def swath_area_attrs_section(area: 'geom.SwathDefinition') -> str: # noqa F821
    """Create html for attribute section based on SwathDefinition.

    Args:
        area : Swath definition.

    Returns:
        Html with collapsible section of swath attributes.

    Todo:
        - Improve resolution estimation from lat/lon arrays. Maybe use CoordinateDefinition.geocentric_resolution?

    """
    if np.ndim(area.lons) == 1:
        area_name = "1D Swath"
        resolution_str = "NAxNA"
        height, width = "NA", "NA"
        area_units = "m"
    else:
        if isinstance(area.lons, np.ndarray) and isinstance(area.lats, np.ndarray):
            # Calculate and estimated resolution from lats/lons in meter
            area_name = "Arbitrary Swath"
            resolution_y = np.mean(area.lats[0:-1, :] - area.lats[1::, :])
            resolution_x = np.mean(area.lons[:, 1::] - area.lons[:, 0:-1])
            resolution = np.mean(np.array([resolution_x, resolution_y]))
            resolution = np.round(40075000 * resolution / 360, 1)
            resolution_str = f"{resolution}x{resolution}"
        else:
            lon_attrs = area.lons.attrs
            lat_attrs = area.lats.attrs

            # use resolution from lat/lons dataarray attributes -> are these always set? -> Maybe try/except?
            area_name = f"{lon_attrs.get('sensor')} Swath"
            resolution_str = "x".join([str(round(x.get("resolution"), 1)) for x in [lat_attrs, lon_attrs]])

        area_units = "m"
        height, width = area.lons.shape

    attrs_icon = _icon("icon-file-text2")

    area_attrs = ("<dl>"
                  # f"<dt>Area name</dt><dd>{area_name}</dd>"
                  f"<dt>Description</dt><dd>{area_name}</dd>"
                  f"<dt>Width/Height</dt><dd>{width}/{height} Pixel</dd>"
                  f"<dt>Resolution XxY (SSP)</dt><dd>{resolution_str} {area_units}</dd>"
                  "</dl>"
                  )

    if xarray and not isinstance(area.lons, np.ndarray):
        ds_dict = {i.attrs['name']: i.rename(i.attrs['name']) for i in [area.lons, area.lats]}
        dss = xr.merge(ds_dict.values())

        area_attrs += _obj_repr(dss, header_components=[""], sections=[datavar_section(dss.data_vars)])
    else:
        with np.printoptions(threshold=50):
            lons = f"{area.lons}".replace("\n", "<br>")
            lats = f"{area.lats}".replace("\n", "<br>")
            area_attrs += ("<div class='xr-wrap', style='display:none'>"
                           "<div class='xr-header'></div>"
                           "<ul class='xr-sections'>"
                           "<li class='xr-section-item'>"
                               "<div class='xr-section-details', style='display:contents'>"  # noqa E127
                                   "<ul class='xr-var-list'>"  # noqa E127
                                       "<li class='xr-var-item'>"  # noqa E127
                                           "<div class='xr-var-name'>"  # noqa E127
                                           "<span>Longitude</span>"
                                           "</div>"
                                           f"<div class=xr-var-preview xr-preview>{lons}</div>"
                                       "</li>"
                                       "<li class='xr-var-item'>"
                                           "<div class='xr-var-name'>"
                                           "<span>Latitude</span>"
                                           "</div>"
                                           f"<div class=xr-var-preview xr-preview>{lats}</div>"
                                       "</li>"
                                   "</ul>"
                               "</div>"
                           "</li>"
                           "</ul>"
                           "</div>"
                           "</div>"
                           )

    coll = collapsible_section("Properties", details=area_attrs, icon=attrs_icon)

    return f"{coll}"


def area_repr(area: Union['geom.AreaDefinition', 'geom.SwathDefinition'],
              include_header: bool = True,
              include_static_files: bool = True,
              map_content: str | None = None,
              ):
    """Return html repr of an AreaDefinition.

    Args:
        area : Area definition.
        include_header : If true a header with object type will be included in
            the html. This is mainly intended for display in Jupyter Notebooks. For the
            display in the overview of area definitions for the Satpy documentation this
            should be set to false.
        include_static_files : Load and include css and html needed for representation.
        map_content : Optionally override the map section contents. Can be any string
            that is valid HTML between a "<div></div>" tag.

    Returns:
        Html.

    """
    if include_static_files:
        icons_svg, css_style = _load_static_files()
        html = f"{icons_svg}<style>{css_style}</style>"
    else:
        html = ""

    obj_type = f"pyresample.{type(area).__name__}"
    header = ("<div class='pyresample-header'>"
              "<div class='pyresample-obj-type'>"
              f"{escape(obj_type)}"
              "</div>"
              "</div>"
              )

    html += (f"<pre class='pyresample-text-repr-fallback'>{escape(repr(area))}</pre>"
             "<div class='pyresample-wrap' style='display:none'>"
             )

    if include_header:
        html += f"{header}"

    html += "<div class='pyresample-area-sections'>"
    if isinstance(area, geom.AreaDefinition):
        html += proj_area_attrs_section(area)
        map_icon = _icon("icon-globe")
        if map_content is None:
            if cartopy:
                map_content = plot_area_def(area, fmt="svg")
            else:
                map_content = "Note: If cartopy is installed a display of the area can be seen here"
        coll = collapsible_section("Map",
                                   details=map_content,
                                   collapsed=True,
                                   icon=map_icon)

        html += str(coll)
    elif isinstance(area, geom.SwathDefinition):
        html += swath_area_attrs_section(area)

    html += "</div>"

    return html
