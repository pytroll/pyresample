#!/usr/bin/env python
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
"""Functions for html representation of area definition."""

import uuid
from functools import lru_cache
from html import escape
from importlib.resources import read_binary

import numpy as np

import pyresample.geometry as geom

try:
    import cartopy
    cart = True
except ModuleNotFoundError:
    cart = False

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


def plot_area_def(area, feature_res="110m", fmt="svg"):
    """Plot area.

    CURRENTLY feature_res is not used instead cartopy auto scaled features are added.

    Args:
        area (Union[:class:`~pyresample.geometry.AreaDefinition`, :class:`~pyresample.geometry.SwathDefinition`])
        feature_res (str):
            Resolution of the features added to the map. Argument is handed over
            to `scale` parameter in cartopy.feature.
        fmt (str): Output format of the plot. The output is the string representation of
            the respective format xml for svg and base64 for png. Either svg (default) or png.
            If other plot is just shown.

    Returns:
        str: svg or png image as string.
    """
    import base64
    from io import BytesIO, StringIO

    import matplotlib.pyplot as plt

    if isinstance(area, geom.AreaDefinition):
        crs = area.to_cartopy_crs()
        fig, ax = plt.subplots(subplot_kw=dict(projection=crs))
    elif isinstance(area, geom.SwathDefinition):
        import cartopy.crs as ccrs
        from shapely.geometry.polygon import Polygon

        lx, ly = area.get_edge_lonlats()

        crs = cartopy.crs.Mercator()
        fig, ax = plt.subplots(subplot_kw=dict(projection=crs))

        poly = Polygon(list(zip(lx, ly)))
        ax.add_geometries([poly], crs=ccrs.CRS(area.crs), facecolor="none", edgecolor="red")
        ax.set_extent(poly.bounds)

    coastlines = cartopy.feature.NaturalEarthFeature(category="physical",
                                                     name="coastline",
                                                     scale=feature_res,
                                                     linewidth=1,
                                                     facecolor="never")
    borders = cartopy.feature.NaturalEarthFeature(category="cultural",
                                                  name="admin_0_boundary_lines_land", # noqa E114
                                                  scale=feature_res,
                                                  edgecolor="black",
                                                  facecolor="never") # noqa E1>
    ocean = cartopy.feature.OCEAN

    ax.add_feature(borders)
    ax.add_feature(coastlines)
    ax.add_feature(ocean, color="lightgrey")

    plt.tight_layout(pad=0)

    if fmt == "svg":
        svg_str = StringIO()
        plt.savefig(svg_str, format="svg", bbox_inches="tight")
        plt.close()
        return svg_str.getvalue()

    elif fmt == "png":
        png_str = BytesIO()
        plt.savefig(png_str, format="png", bbox_inches="tight")
        img_str = f"<img src='data:image/png;base64, {base64.encodestring(png_str.getvalue()).decode('utf-8')}'/>"
        plt.close()
        return img_str

    else:
        plt.show()


def collapsible_section(name, inline_details="", details="", enabled=True, collapsed=False, icon=None):
    """Create a collapsible section.

    Args:
      name (str): Name of the section
      inline_details (str): Information to show when section is collapsed. Default nothing.
      details (str): Details to show when section is expanded.
      enabled (boolean): Is collapsing enabled. Default True.
      collapsed (boolean): Is the section collapsed on first show. Default False.

    Returns:
      str: Html div structure for collapsible section.

    """
    # "unique" id to expand/collapse the section
    data_id = "section-" + str(uuid.uuid4())

    enabled = "" if enabled else "disabled"
    collapsed = "" if collapsed else "checked"
    tip = " title='Expand/collapse section'" if enabled else ""

    if icon is None:
        icon = _icon("icon-database")

    return ("<div class='pyresample-area-section-item'>"
            f"<input id='{data_id}' class='pyresample-area-section-in' "
            f"type='checkbox' {enabled} {collapsed}>"
            f"<label for='{data_id}' {tip}>{icon} {name}</label>"
            f"<div class='pyresample-area-section-preview'>{inline_details}</div>"
            f"<div class='pyresample-area-section-details'>{details}</div>"
            "</div>"
            )


def map_section(area):
    """Create html for map section.

    Args:
        area(Union[:class:`~pyresample.geometry.AreaDefinition`, :class:`~pyresample.geometry.SwathDefinition`]):
            Area definition or Swath definition.

    Returns:
        str: String of html.

    """
    map_icon = _icon("icon-globe")

    if cart:
        coll = collapsible_section("Map", details=plot_area_def(area), collapsed=True, icon=map_icon)
    else:
        coll = collapsible_section("Map",
                                   details="Note: If cartopy is installed a display of the area can be seen here",
                                   collapsed=True, icon=map_icon)

    return f"{coll}"


def proj_area_attrs_section(area):
    """Create html for attribute section based on an area Area.

    Args:
        area (:class:`~pyresample.geometry.AreaDefinition`):
            Area definition.

    Returns:
        str: String of html.

    """
    resolution_str = "/".join([str(round(x, 1)) for x in area.resolution])
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
                  f"<dd>{tuple(round(x, 4) for x in area.area_extent)}</dd>"
                  "</dl>"
                  )

    coll = collapsible_section("Properties", details=area_attrs, icon=attrs_icon)

    return f"{coll}"


def swath_area_attrs_section(area):
    """Create html for attribute section based on SwathDefinition.

    Args:
        area (:class:`~pyresample.geometry.SwathDefinition`): Swath definition.

    Returns:
        str: String of html.

    """
    if isinstance(area.lons, np.ndarray) & isinstance(area.lats, np.ndarray):
        area_name = "Area name"
        resolution_y = np.mean(area.lats[0:-1, :] - area.lats[1::, :])
        resolution_x = np.mean(area.lons[:, 1::] - area.lons[:, 0:-1])
        resolution = np.mean(np.array([resolution_x, resolution_y]))
        resolution = 40075000 * resolution / 360
        resolution_str = f"{resolution}/{resolution}"
        area_units = "m"
    else:
        lon_attrs = area.lons.attrs
        lat_attrs = area.lats.attrs

        area_name = f"{lon_attrs.get('sensor')} swath"
        resolution_str = "/".join([str(round(x.get("resolution"), 1)) for x in [lat_attrs, lon_attrs]])
        area_units = "m"

    height, width = area.lons.shape

    attrs_icon = _icon("icon-file-text2")

    area_attrs = ("<dl>"
                  # f"<dt>Area name</dt><dd>{area_name}</dd>"
                  f"<dt>Description</dt><dd>{area_name}</dd>"
                  f"<dt>Width/Height</dt><dd>{width}/{height} Pixel</dd>"
                  f"<dt>Resolution x/y (SSP)</dt><dd>{resolution_str} {area_units}</dd>"
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


def area_repr(area, include_header=True, include_static_files=True):
    """Return html repr of an AreaDefinition.

    Args:
        area (Union[:class:`~pyresample.geometry.AreaDefinition`, :class:`~pyresample.geometry.AreaDefinition`]):
            Area definition.
        include_header (Optional[bool]): If true a header with object type will be included in
            the html. This is mainly intented for display in Jupyter Notebooks. For the
            display in the overview of area definitions for the Satpy documentation this
            should be set to false.
        include_static_files (Optional[bool]): Load and include css and html needed for representation.

    Returns:
        str: String of html.

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
        html += map_section(area)
    elif isinstance(area, geom.SwathDefinition):
        html += swath_area_attrs_section(area)

    html += "</div>"

    return html