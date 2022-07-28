#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    crs = area_def.to_cartopy_crs()
    fig, ax = plt.subplots(subplot_kw=dict(projection=crs))#, figsize=(10,10))
    #coastlines = ax.coastlines(resolution="50m")
    #high_res_borders = cartopy.feature.NaturalEarthFeature(category="cultural",
                                                   #name="admin_0_boundary_lines_land", # noqa E114
                                                   #scale=feature_res, edgecolor="black", facecolor="never") # noqa E1>
    #ax.add_feature(high_res_borders)
    ax.add_feature(BORDERS)
    ax.add_feature(COASTLINE)
    
    #ax.annotate(area.area_extent[0:2], xy=area.area_extent[0:2], color="red", xycoords=ccrs.Geostationary()._as_mpl_transform(ax), ha="right", va="top")
    #ax.annotate(area.area_extent[2:4], xy=area.area_extent[2:4], color="red", xycoords=ccrs.Geostationary()._as_mpl_transform(ax), ha="left", va="bottom")
    #ax.annotate(area.area_extent[2:4], xy=[1885700, 5000000], color="red", xycoords=ccrs.Geostationary()._as_mpl_transform(ax), ha="left", va="bottom")
    ax.annotate(np.round(area_def.area_extent[2:4]), xy=[1885700, 5000000], xycoords=ccrs.Geostationary()._as_mpl_transform(ax), xytext=[1, 1], color="red", textcoords="axes fraction", ha="center", va="bottom")
    
    ax.set_global()
    plt.tight_layout()
    #img = plt.imshow(result, transform=crs, extent=crs.bounds, origin='upper')
    #cbar = plt.colorbar()
    if file is not None:
        plt.savefig(file)
    else:
        plt._repr_html_()


def area_repr(areadefinition):
    """Return html repr of an AreaDefinition."""
    icons_svg, css_style = _load_static_files()

    obj_type = f"pyresample.{type(areadefinition).__name__}"
    header = ("<div class='pyresample-header'>"
              "<div class='pyresample-obj-type'>"
              f"{escape(obj_type)}"
              "</div>"
              "</div>"
             )

    resolution_str = "/".join([str(round(x, 1)) for x in areadefinition.resolution])
    area_units = areadefinition.proj_dict["units"]
    area_attrs = ("<dl>"
           f"<dt>Area name</dt><dd>{areadefinition.area_id}</dd>"
           f"<dt>Description</dt><dd>{areadefinition.description}</dd>"
           f"<dt>Width/Height</dt><dd>{areadefinition.width}/{areadefinition.height} Pixel</dd>"
           f"<dt>Resolution x/y</dt><dd>{resolution_str} {area_units}</dd>"
           f"<dt>Extent</dt><dd>{tuple(round(x, 4) for x in areadefinition.area_extent)}</dd>"
           "</dl>"
           ) 

    # area_plot = plot_area_def(areadefinition)

    html = ("<div>"
           f"{icons_svg}<style>{css_style}</style>"
           f"<pre class='pyresample-text-repr-fallback'>{escape(repr(areadefinition))}</pre>"
            "<div class='pyresample-wrap' style='display:none'>"
           f"{header}"
           f"<div class='pyresample-area'>"
           f"<div class='pyresample-area-attrs'>{area_attrs}</div>"
           f"<div class='pyresample-area-plot'>{plot_area_def(areadefinition)}</div>"
           "</div>"
           )

    return html
