#!/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid
from functools import lru_cache, partial
from html import escape
from importlib.resources import read_binary

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


def plot_area_def(area_def, feature_res="110m", file=None):
    """Plot area definition.

    CURRENTLY feature_res is not used instead cartopy auto scaled features are added.

    Parameters
    ----------
    area_def : pyresample.AreaDefinition
    feature_res : str
        Resolution of the features added to the map. Argument is handed over
        to `scale` parameter in cartopy.feature.
    file : str
        Filename to save to the plot to as png. Defaults to None which shows
        plot.
    """
    import matplotlib.pyplot as plt
    #import cartopy
    from cartopy.feature import BORDERS
    from cartopy.feature import COASTLINE
    import cartopy.crs as ccrs
    import numpy as np
    from io import StringIO, BytesIO
    import base64

    plt.ioff()
    
    fmt = "svg"

    crs = area_def.to_cartopy_crs()
    fig, ax = plt.subplots(subplot_kw=dict(projection=crs))

    # coastlines = ax.coastlines(resolution="50m", color='black', linewidth=1)
    #high_res_borders = cartopy.feature.NaturalEarthFeature(category="cultural",
                                                   #name="admin_0_boundary_lines_land", # noqa E114
                                                   #scale=feature_res, edgecolor="black", facecolor="never") # noqa E1>
    #ax.add_feature(high_res_borders)
    ax.add_feature(BORDERS)
    ax.add_feature(COASTLINE)
    
    #ax.annotate(area.area_extent[0:2], xy=area.area_extent[0:2], color="red", xycoords=ccrs.Geostationary()._as_mpl_transform(ax), ha="right", va="top")
    #ax.annotate(area.area_extent[2:4], xy=area.area_extent[2:4], color="red", xycoords=ccrs.Geostationary()._as_mpl_transform(ax), ha="left", va="bottom")
    #ax.annotate(area.area_extent[2:4], xy=[1885700, 5000000], color="red", xycoords=ccrs.Geostationary()._as_mpl_transform(ax), ha="left", va="bottom")
    # add bounding box coordinates to edges of plot
    # ax.annotate(np.round(area_def.area_extent[2:4]), xy=[1885700, 5000000], xycoords=ccrs.Geostationary()._as_mpl_transform(ax), xytext=[1, 1], color="red", textcoords="axes fraction", ha="center", va="bottom")
    
    # ax.set_global()
    plt.tight_layout(pad=0)

    if file is not None:
        plt.savefig(file)
    elif fmt=="svg":
        svg_str = StringIO()
        plt.savefig(svg_str, format="svg", bbox_inches="tight")
        return svg_str.getvalue()
    elif fmt=="png":
        png_str = BytesIO()
        plt.savefig(png_str, format="png", bbox_inches="tight")
        img_str = f"<img src='data:image/png;base64, {base64.encodestring(png_str.getvalue()).decode('utf-8')}'/>"

        return img_str

def collapsible_section(name, inline_details="", details="", enabled=True, collapsed=False, icon=None):
    """Creates a collapsible section.

    Args:
      name (str):
      inline_details (str):
      details (str):
      n_items (??):
      enables (boolean):
      collapsed (boolean):

    Returns:
      str:

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
    """Creates html for map section."""
    map_icon = _icon("icon-globe")

    coll = collapsible_section("Map", details=plot_area_def(areadefinition), collapsed=True, icon=map_icon)

    return f"{coll}"


def attrs_section(areadefinition):
    """Creates html for attribute section."""
    resolution_str = "/".join([str(round(x, 1)) for x in areadefinition.resolution])
    proj_dict = areadefinition.proj_dict
    proj_str = "{{{}}}".format(", ".join(["'%s': '%s'" % (str(k), str(proj_dict[k])) for k in sorted(proj_dict.keys())]))
    try:
        area_units = proj_dict["units"]
    except:
        area_units = ""

    attrs_icon = _icon("icon-file-text2")

    area_attrs = ("<dl>"
           f"<dt>Area name</dt><dd>{areadefinition.area_id}</dd>"
           f"<dt>Description</dt><dd>{areadefinition.description}</dd>"
           f"<dt>Projection</dt><dd>{proj_str}</dd>"
           f"<dt>Width/Height</dt><dd>{areadefinition.width}/{areadefinition.height} Pixel</dd>"
           f"<dt>Resolution x/y</dt><dd>{resolution_str} {area_units}</dd>"
           f"<dt>Extent (ll_x, ll_y, ur_x, ur_y)</dt><dd>{tuple(round(x, 4) for x in areadefinition.area_extent)}</dd>"
           "</dl>"
           ) 

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


    html = (#"<div>"
           f"{icons_svg}<style>{css_style}</style>"
           f"<pre class='pyresample-text-repr-fallback'>{escape(repr(areadefinition))}</pre>"
            "<div class='pyresample-wrap' style='display:none'>"
           )

    if include_header:
        html += f"{header}"

    html += "<div class='pyresample-area-sections'>"
    html += attrs_section(areadefinition)

    html += map_section(areadefinition)

    html += "</div>" #"</div>"

    return html
