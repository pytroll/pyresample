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
"""Functions for html representation of area definition."""

import uuid
from functools import lru_cache
from html import escape
from importlib.resources import read_binary
import pyresample.geometry as geom
from cartopy.crs import CRS
import numpy as np
import xarray as xr
from xarray.core.formatting_html import _obj_repr, datavar_section

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


def plot_area_def(area_def, feature_res="110m", fmt="svg"):
    """Plot area definition.

    CURRENTLY feature_res is not used instead cartopy auto scaled features are added.

    Args:
        area_def : pyresample.AreaDefinition
        feature_res : str
            Resolution of the features added to the map. Argument is handed over
            to `scale` parameter in cartopy.feature.
        fmt (str): Output format of the plot. The output is the string representation of
            the respective format xml for svg and base64 for png. Either svg (default) or png.
            If other plot is just shown.
    """
    import base64
    from io import BytesIO, StringIO

    import cartopy
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from pyresample.kd_tree import resample_nearest

    if isinstance(area_def, geom.AreaDefinition):
        crs = area_def.to_cartopy_crs()
        fig, ax = plt.subplots(subplot_kw=dict(projection=crs))
    elif isinstance(area_def, geom.SwathDefinition):
        bb_area = area_def.compute_optimal_bb_area()
        crs = bb_area.to_cartopy_crs()
        fig, ax = plt.subplots(subplot_kw=dict(projection=crs))

        polygon = False
        if polygon:
            from pygeos import polygons
            import geopandas as gpd
            import pandas as pd

            lx, ly = area_def.get_edge_lonlats()
            poly = polygons(list(zip(lx, ly)))

            df = pd.DataFrame({"area": ["modis_swath"], "geom": [poly]})
            gdf = gpd.GeoDataFrame(df, crs=area_def.crs)
            gdf = gdf.set_geometry("geom")
            gdf = gdf.to_crs(crs.proj4_init)
            # gdf.plot(ax=ax)
            ax.add_geometries(gdf["geom"], crs=crs)
        else:
	    # bb_area = area_def.compute_optimal_bb_area()
	    # crs = bb_area.to_cartopy_crs()
	    # fig, ax = plt.subplots(subplot_kw=dict(projection=crs))

            data = np.ones_like(area_def.lons)
            data = resample_nearest(area_def, data, bb_area, radius_of_influence=20000, fill_value=None)

            cmap = colors.ListedColormap(["white", "grey"])
            bounds = [0, 0.5, 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)

            ax.imshow(data, transform=crs, extent=crs.bounds, origin="upper", cmap=cmap, norm=norm, alpha=0.35)

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
    ax.add_feature(borders)
    ax.add_feature(coastlines)

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


def map_section(areadefinition):
    """Create html for map section.

    Args:
        areadefinition (:class:`~pyresample.geometry.AreaDefinition`): Area definition.
        include_header (boolean): If true a header with object type will be included in
            the html. This is mainly intented for display in Jupyter Notebooks. For the
            display in the overview of area definitions for the Satpy documentation this
            should be set to false.

    Returns:
        str: String of html.

    """
    map_icon = _icon("icon-globe")

    coll = collapsible_section("Map", details=plot_area_def(areadefinition), collapsed=True, icon=map_icon)

    return f"{coll}"


def proj_area_attrs_section(areadefinition):
    """Create html for attribute section based on AreaDefinition.

    Args:
        areadefinition (:class:`~pyresample.geometry.AreaDefinition`): Area definition.
        include_header (boolean): If true a header with object type will be included in
            the html. This is mainly intented for display in Jupyter Notebooks. For the
            display in the overview of area definitions for the Satpy documentation this
            should be set to false.

    Returns:
        str: String of html.

    """
    resolution_str = "/".join([str(round(x, 1)) for x in areadefinition.resolution])
    proj_dict = areadefinition.proj_dict
    proj_str = "{{{}}}".format(", ".join(["'%s': '%s'" % (str(k), str(proj_dict[k])) for k in
                                          sorted(proj_dict.keys())]))
    area_units = proj_dict.get("units", "")

    attrs_icon = _icon("icon-file-text2")

    area_attrs = ("<dl>"
                  f"<dt>Area name</dt><dd>{areadefinition.area_id}</dd>"
                  f"<dt>Description</dt><dd>{areadefinition.description}</dd>"
                  f"<dt>Projection</dt><dd>{proj_str}</dd>"
                  f"<dt>Width/Height</dt><dd>{areadefinition.width}/{areadefinition.height} Pixel</dd>"
                  f"<dt>Resolution x/y (SSP)</dt><dd>{resolution_str} {area_units}</dd>"
                  f"<dt>Extent (ll_x, ll_y, ur_x, ur_y)</dt>"
                  f"<dd>{tuple(round(x, 4) for x in areadefinition.area_extent)}</dd>"
                  "</dl>"
                  )

    coll = collapsible_section("Properties", details=area_attrs, icon=attrs_icon)

    return f"{coll}"


def swath_area_attrs_section(areadefinition):
    """Create html for attribute section based on SwathDefinition.

    Args:
        areadefinition (:class:`~pyresample.geometry.SwathDefinition`): Swath definition.

    Returns:
        str: String of html.

    """
    lon_attrs = areadefinition.lons.attrs
    lat_attrs = areadefinition.lats.attrs

    area_name = f"{lon_attrs.get('sensor')} swath"
    height, width = areadefinition.lons.shape
    resolution_str = "/".join([str(round(x.get("resolution"), 1)) for x in [lat_attrs, lon_attrs]])
    area_units = "m"

    attrs_icon = _icon("icon-file-text2")

    area_attrs = ("<dl>"
                  # f"<dt>Area name</dt><dd>{area_name}</dd>"
                  f"<dt>Description</dt><dd>{area_name}</dd>"
                  f"<dt>Width/Height</dt><dd>{width}/{height} Pixel</dd>"
                  f"<dt>Resolution x/y (SSP)</dt><dd>{resolution_str} {area_units}</dd>"
                  "</dl>"
                  )

    ds_dict = {i.attrs['name']: i.rename(i.attrs['name']) for i in [areadefinition.lons, areadefinition.lats]}
    dss = xr.merge(ds_dict.values())

    area_attrs += _obj_repr(dss, header_components=[""], sections=[datavar_section(dss.data_vars)])

    coll = collapsible_section("Properties", details=area_attrs, icon=attrs_icon)

    return f"{coll}"


def area_repr(areadefinition, include_header=True):
    """Return html repr of an AreaDefinition.

    Args:
        areadefinition (:class:`~pyresample.geometry.AreaDefinition`): Area definition.
        include_header (boolean): If true a header with object type will be included in
            the html. This is mainly intented for display in Jupyter Notebooks. For the
            display in the overview of area definitions for the Satpy documentation this
            should be set to false.

    Returns:
        str: String of html.

    """
    icons_svg, css_style = _load_static_files()

    obj_type = f"pyresample.{type(areadefinition).__name__}"
    header = ("<div class='pyresample-header'>"
              "<div class='pyresample-obj-type'>"
              f"{escape(obj_type)}"
              "</div>"
              "</div>"
              )

    html = (f"{icons_svg}<style>{css_style}</style>"
            f"<pre class='pyresample-text-repr-fallback'>{escape(repr(areadefinition))}</pre>"
            "<div class='pyresample-wrap' style='display:none'>"
            )

    if include_header:
        html += f"{header}"

    html += "<div class='pyresample-area-sections'>"
    if isinstance(areadefinition, geom.AreaDefinition):
        html += proj_area_attrs_section(areadefinition)
    elif isinstance(areadefinition, geom.SwathDefinition):
        html += swath_area_attrs_section(areadefinition)

    html += map_section(areadefinition)

    html += "</div>"

    return html
