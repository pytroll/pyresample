#!/usr/bin/env python
# encoding: utf8
#
# Copyright (C) 2010-2018
#
# Authors:
#    Esben S. Nielsen
#    Thomas Lavergne
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

from __future__ import absolute_import
import numpy as np


def ellps2axis(ellps_name):
    """Get semi-major and semi-minor axis from ellipsis definition

    Parameters
    ---------
    ellps_name : str
        Standard name of ellipsis

    Returns
    -------
    (a, b) : semi-major and semi-minor axis
    """

    ellps = {'helmert': {'a': 6378200.0, 'b': 6356818.1696278909},
             'intl': {'a': 6378388.0, 'b': 6356911.9461279465},
             'merit': {'a': 6378137.0, 'b': 6356752.2982159676},
             'wgs72': {'a': 6378135.0, 'b': 6356750.5200160937},
             'sphere': {'a': 6370997.0, 'b': 6370997.0},
             'clrk66': {'a': 6378206.4000000004, 'b': 6356583.7999999998},
             'nwl9d': {'a': 6378145.0, 'b': 6356759.7694886839},
             'lerch': {'a': 6378139.0, 'b': 6356754.2915103417},
             'evrstss': {'a': 6377298.5559999999, 'b': 6356097.5503008962},
             'evrst30': {'a': 6377276.3449999997, 'b': 6356075.4131402401},
             'mprts': {'a': 6397300.0, 'b': 6363806.2827225132},
             'krass': {'a': 6378245.0, 'b': 6356863.0187730473},
             'walbeck': {'a': 6376896.0, 'b': 6355834.8466999996},
             'kaula': {'a': 6378163.0, 'b': 6356776.9920869097},
             'wgs66': {'a': 6378145.0, 'b': 6356759.7694886839},
             'evrst56': {'a': 6377301.2429999998, 'b': 6356100.2283681016},
             'new_intl': {'a': 6378157.5, 'b': 6356772.2000000002},
             'airy': {'a': 6377563.3959999997, 'b': 6356256.9100000001},
             'bessel': {'a': 6377397.1550000003, 'b': 6356078.9628181886},
             'seasia': {'a': 6378155.0, 'b': 6356773.3205000004},
             'aust_sa': {'a': 6378160.0, 'b': 6356774.7191953054},
             'wgs84': {'a': 6378137.0, 'b': 6356752.3142451793},
             'hough': {'a': 6378270.0, 'b': 6356794.3434343431},
             'wgs60': {'a': 6378165.0, 'b': 6356783.2869594367},
             'engelis': {'a': 6378136.0499999998, 'b': 6356751.3227215428},
             'apl4.9': {'a': 6378137.0, 'b': 6356751.796311819},
             'andrae': {'a': 6377104.4299999997, 'b': 6355847.4152333336},
             'sgs85': {'a': 6378136.0, 'b': 6356751.301568781},
             'delmbr': {'a': 6376428.0, 'b': 6355957.9261637237},
             'fschr60m': {'a': 6378155.0, 'b': 6356773.3204827355},
             'iau76': {'a': 6378140.0, 'b': 6356755.2881575283},
             'plessis': {'a': 6376523.0, 'b': 6355863.0},
             'cpm': {'a': 6375738.7000000002, 'b': 6356666.221912113},
             'fschr68': {'a': 6378150.0, 'b': 6356768.3372443849},
             'mod_airy': {'a': 6377340.1890000002, 'b': 6356034.4460000005},
             'grs80': {'a': 6378137.0, 'b': 6356752.3141403561},
             'bess_nam': {'a': 6377483.8650000002, 'b': 6356165.3829663256},
             'fschr60': {'a': 6378166.0, 'b': 6356784.2836071067},
             'clrk80': {'a': 6378249.1449999996, 'b': 6356514.9658284895},
             'evrst69': {'a': 6377295.6639999999, 'b': 6356094.6679152036},
             'grs67': {'a': 6378160.0, 'b': 6356774.5160907144},
             'evrst48': {'a': 6377304.0630000001, 'b': 6356103.0389931547}}
    try:
        ellps_axis = ellps[ellps_name.lower()]
        a = ellps_axis['a']
        b = ellps_axis['b']
    except KeyError as e:
        raise ValueError(('Could not determine semi-major and semi-minor axis '
                          'of specified ellipsis %s') % ellps_name)
    return a, b


def area_def2basemap(area_def, **kwargs):
    """Get Basemap object from AreaDefinition

    Parameters
    ---------
    area_def : object
        geometry.AreaDefinition object
    \*\*kwargs: Keyword arguments
        Additional initialization arguments for Basemap

    Returns
    -------
    bmap : Basemap object
    """

    import warnings
    warnings.warn("Basemap is no longer maintained. Please switch to cartopy "
                  "by using 'area_def.to_cartopy_crs()'. See the pyresample "
                  "documentation for more details.", DeprecationWarning)

    from mpl_toolkits.basemap import Basemap
    try:
        a, b = ellps2axis(area_def.proj_dict['ellps'])
        rsphere = (a, b)
    except KeyError:
        try:
            a = float(area_def.proj_dict['a'])
            try:
                b = float(area_def.proj_dict['b'])
                rsphere = (a, b)
            except KeyError:
                rsphere = a
        except KeyError:
            # Default to WGS84 ellipsoid
            a, b = ellps2axis('wgs84')
            rsphere = (a, b)

    # Add projection specific basemap args to args passed to function
    basemap_args = kwargs
    basemap_args['rsphere'] = rsphere

    if area_def.proj_dict['proj'] in ('ortho', 'geos', 'nsper'):
        llcrnrx, llcrnry, urcrnrx, urcrnry = area_def.area_extent
        basemap_args['llcrnrx'] = llcrnrx
        basemap_args['llcrnry'] = llcrnry
        basemap_args['urcrnrx'] = urcrnrx
        basemap_args['urcrnry'] = urcrnry
    else:
        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = area_def.area_extent_ll
        basemap_args['llcrnrlon'] = llcrnrlon
        basemap_args['llcrnrlat'] = llcrnrlat
        basemap_args['urcrnrlon'] = urcrnrlon
        basemap_args['urcrnrlat'] = urcrnrlat

    if area_def.proj_dict['proj'] == 'eqc':
        basemap_args['projection'] = 'cyl'
    else:
        basemap_args['projection'] = area_def.proj_dict['proj']

    # Try adding potentially remaining args
    for key in ('lon_0', 'lat_0', 'lon_1', 'lat_1', 'lon_2', 'lat_2',
                'lat_ts'):
        try:
            basemap_args[key] = float(area_def.proj_dict[key])
        except KeyError:
            pass

    return Basemap(**basemap_args)


def _basemap_get_quicklook(area_def, data, vmin=None, vmax=None,
                           label='Variable (units)', num_meridians=45,
                           num_parallels=10, coast_res='110m', cmap='jet'):
    if area_def.shape != data.shape:
        raise ValueError('area_def shape %s does not match data shape %s' %
                         (list(area_def.shape), list(data.shape)))
    import matplotlib.pyplot as plt
    bmap = area_def2basemap(area_def, resolution=coast_res)
    bmap.drawcoastlines()
    if num_meridians > 0:
        bmap.drawmeridians(np.arange(-180, 180, num_meridians))
    if num_parallels > 0:
        bmap.drawparallels(np.arange(-90, 90, num_parallels))
    if not (np.ma.isMaskedArray(data) and data.mask.all()):
        col = bmap.imshow(data, origin='upper', vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(col, shrink=0.5, pad=0.05).set_label(label)
    return plt


def _get_quicklook(area_def, data, vmin=None, vmax=None,
                   label='Variable (units)', num_meridians=45,
                   num_parallels=10, coast_res='110m', cmap='jet'):
    """Get default cartopy matplotlib plot."""
    bmap_to_cartopy_res = {
        'c': '110m',
        'l': '110m',
        'i': '50m',
        'h': '10m',
        'f': '10m'
    }

    try:
        from pyresample import _cartopy  # noqa
    except ImportError:
        if coast_res.endswith('m'):
            _rev_map = {v: k for k, v in bmap_to_cartopy_res.items()}
            coast_res = _rev_map[coast_res]
        return _basemap_get_quicklook(
            area_def, data, vmin, vmax, label, num_meridians,
            num_parallels, coast_res=coast_res, cmap=cmap)

    if coast_res and coast_res not in ['110m', '50m', '10m']:
        import warnings
        warnings.warn("'coast_res' should be either '110m', '50m', '10m'.")
        coast_res = {
            'c': '110m',
            'l': '110m',
            'i': '50m',
            'h': '10m',
            'f': '10m'
        }[coast_res]

    if area_def.shape != data.shape:
        raise ValueError('area_def shape %s does not match data shape %s' %
                         (list(area_def.shape), list(data.shape)))
    import matplotlib.pyplot as plt
    crs = area_def.to_cartopy_crs()
    ax = plt.axes(projection=crs)
    ax.coastlines(resolution=coast_res)
    ax.set_global()

    xlocs = None
    ylocs = None
    if num_meridians:
        xlocs = np.arange(-180, 180, num_meridians)
    if num_parallels:
        ylocs = np.arange(-90, 90, num_parallels)
    ax.gridlines(xlocs=xlocs, ylocs=ylocs)
    if not (np.ma.isMaskedArray(data) and data.mask.all()):
        col = ax.imshow(data, transform=crs, extent=crs.bounds,
                        origin='upper', vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(col, shrink=0.5, pad=0.05).set_label(label)
    return plt


def show_quicklook(area_def, data, vmin=None, vmax=None,
                   label='Variable (units)', num_meridians=45,
                   num_parallels=10, coast_res='110m', cmap='jet'):
    """Display default quicklook plot

    Parameters
    ---------
    area_def : object
        geometry.AreaDefinition object
    data : numpy array | numpy masked array
        2D array matching area_def. Use masked array for transparent values
    vmin : float, optional
        Min value for luminescence scaling
    vmax : float, optional
        Max value for luminescence scaling
    label : str, optional
        Label for data
    num_meridians : int, optional
        Number of meridians to plot on the globe
    num_parallels : int, optional
        Number of parallels to plot on the globe
    coast_res : {'c', 'l', 'i', 'h', 'f'}, optional
        Resolution of coastlines

    Returns
    -------
    bmap : Basemap object
    """

    plt = _get_quicklook(area_def, data, vmin=vmin, vmax=vmax,
                         label=label, num_meridians=num_meridians,
                         num_parallels=num_parallels, coast_res=coast_res,
                         cmap=cmap)
    plt.show()
    plt.close()


def save_quicklook(filename, area_def, data, vmin=None, vmax=None,
                   label='Variable (units)', num_meridians=45,
                   num_parallels=10, coast_res='110m', backend='AGG',
                   cmap='jet'):
    """Display default quicklook plot

    Parameters
    ----------
    filename : str
        path to output file
    area_def : object
        geometry.AreaDefinition object
    data : numpy array | numpy masked array
        2D array matching area_def. Use masked array for transparent values
    vmin : float, optional
        Min value for luminescence scaling
    vmax : float, optional
        Max value for luminescence scaling
    label : str, optional
        Label for data
    num_meridians : int, optional
        Number of meridians to plot on the globe
    num_parallels : int, optional
        Number of parallels to plot on the globe
    coast_res : {'c', 'l', 'i', 'h', 'f'}, optional
        Resolution of coastlines
    backend : str, optional
        matplotlib backend to use'
    """

    import matplotlib
    matplotlib.use(backend, warn=False)
    plt = _get_quicklook(area_def, data, vmin=vmin, vmax=vmax,
                         label=label, num_meridians=num_meridians,
                         num_parallels=num_parallels, coast_res=coast_res)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
