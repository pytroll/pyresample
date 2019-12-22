## Version v1.14.0 (2019/12/22)

### Issues Closed

* [Issue 242](https://github.com/pytroll/pyresample/issues/242) - AreaDefinition.get_lonlats ignores dtype option ([PR 243](https://github.com/pytroll/pyresample/pull/243))
* [Issue 233](https://github.com/pytroll/pyresample/issues/233) - get_neighbour_info can not handle SwathDefinitions with lat lon of type integers ([PR 235](https://github.com/pytroll/pyresample/pull/235))
* [Issue 229](https://github.com/pytroll/pyresample/issues/229) - Update old documentation on easy quicklook display (stop using rainbow color map!) ([PR 230](https://github.com/pytroll/pyresample/pull/230))
* [Issue 228](https://github.com/pytroll/pyresample/issues/228) - Area definition boundaries where space pixels are excluded

In this release 4 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 245](https://github.com/pytroll/pyresample/pull/245) - Remove pyximports from gradient search
* [PR 243](https://github.com/pytroll/pyresample/pull/243) - Respect dtype when get_lonlats provide dask array ([242](https://github.com/pytroll/pyresample/issues/242))
* [PR 241](https://github.com/pytroll/pyresample/pull/241) - Fix typo in ImageContainerQuick and ImageContainerNearest docs
* [PR 236](https://github.com/pytroll/pyresample/pull/236) - Fix compatibility with pyproj 2.4.2
* [PR 227](https://github.com/pytroll/pyresample/pull/227) - Fix EWA resampling hanging when geolocation had a lot of NaNs
* [PR 224](https://github.com/pytroll/pyresample/pull/224) - Fix deprecation warning for abc classes

#### Features added

* [PR 230](https://github.com/pytroll/pyresample/pull/230) - No rainbow update documentation ([229](https://github.com/pytroll/pyresample/issues/229))
* [PR 225](https://github.com/pytroll/pyresample/pull/225) - Add smarter default radius_of_influence to XArrayResamplerNN resampling
* [PR 222](https://github.com/pytroll/pyresample/pull/222) - Make the uniform shape computation more effective for dask arrays
* [PR 191](https://github.com/pytroll/pyresample/pull/191) - Implement gradient search resampling method

#### Documentation changes

* [PR 241](https://github.com/pytroll/pyresample/pull/241) - Fix typo in ImageContainerQuick and ImageContainerNearest docs
* [PR 238](https://github.com/pytroll/pyresample/pull/238) - Update load_area docstring to mention that multiple files are allowed
* [PR 230](https://github.com/pytroll/pyresample/pull/230) - No rainbow update documentation ([229](https://github.com/pytroll/pyresample/issues/229))

In this release 13 pull requests were closed.


## Version 1.13.2 (2019/10/08)

### Issues Closed

* [Issue 220](https://github.com/pytroll/pyresample/issues/220) - Problem with dynamic areas on numpy arrays with newest pyresample ([PR 221](https://github.com/pytroll/pyresample/pull/221))
* [Issue 148](https://github.com/pytroll/pyresample/issues/148) - Complete dask conversion of XArrayResamplerBilinear
* [Issue 10](https://github.com/pytroll/pyresample/issues/10) - Computing density_of_x (alternatively "counting number of x)" while re-gridding

In this release 3 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 221](https://github.com/pytroll/pyresample/pull/221) - Make optimal bb computation work with numpy arrays ([220](https://github.com/pytroll/pyresample/issues/220))

In this release 1 pull request was closed.


## Version 1.13.1 (2019/09/26)

### Pull Requests Merged

#### Bugs fixed

* [PR 218](https://github.com/pytroll/pyresample/pull/218) - Fix proj_str returning invalid PROJ strings when towgs84 was included
* [PR 217](https://github.com/pytroll/pyresample/pull/217) - Fix get_geostationary_angle_extent assuming a/b definitions
* [PR 216](https://github.com/pytroll/pyresample/pull/216) - Fix proj4 radius parameters for spherical cases

In this release 3 pull requests were closed.

## Version 1.13.0 (2019/09/13)

### Issues Closed

* [Issue 210](https://github.com/pytroll/pyresample/issues/210) - Incompatibility with new proj/pyproj versions

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 213](https://github.com/pytroll/pyresample/pull/213) - Remove extra conversion to dask array
* [PR 208](https://github.com/pytroll/pyresample/pull/208) - Bugfix bilinear resampler masking ([735](https://github.com/pytroll/satpy/issues/735))
* [PR 207](https://github.com/pytroll/pyresample/pull/207) - Make output index tiling in bilinear interpolation work with dask
* [PR 205](https://github.com/pytroll/pyresample/pull/205) - Exclude NaNs from Bucket Average
* [PR 197](https://github.com/pytroll/pyresample/pull/197) - Fix to_cartopy_crs for latlong projections
* [PR 196](https://github.com/pytroll/pyresample/pull/196) - Improve handling of EPSG codes with pyproj 2.0+

#### Features added

* [PR 212](https://github.com/pytroll/pyresample/pull/212) - Use slices in bilinear resampler
* [PR 203](https://github.com/pytroll/pyresample/pull/203) - Add Numpy version limitation for Python 2
* [PR 198](https://github.com/pytroll/pyresample/pull/198) - Clarify warning if no overlap data and projection
* [PR 196](https://github.com/pytroll/pyresample/pull/196) - Improve handling of EPSG codes with pyproj 2.0+
* [PR 192](https://github.com/pytroll/pyresample/pull/192) - Add bucket resampling

#### Documentation changes

* [PR 204](https://github.com/pytroll/pyresample/pull/204) - Add Example for Regular Lat-Lon Grid
* [PR 201](https://github.com/pytroll/pyresample/pull/201) - fix bug in plot example code
* [PR 198](https://github.com/pytroll/pyresample/pull/198) - Clarify warning if no overlap data and projection
* [PR 195](https://github.com/pytroll/pyresample/pull/195) - Update docs for create_area_def and improve AreaDefinition property consistency

In this release 15 pull requests were closed.


## Version 1.12.3 (2019/05/17)

### Pull Requests Merged

#### Bugs fixed

* [PR 193](https://github.com/pytroll/pyresample/pull/193) - Fix striding slicing in AreaDefinition

In this release 1 pull request was closed.


## Version 1.12.2 (2019/05/10)

### Issues Closed

* [Issue 187](https://github.com/pytroll/pyresample/issues/187) - Numerous `RuntimeWarning`s when resampling

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 190](https://github.com/pytroll/pyresample/pull/190) - Fix aggregate method using non-serializable internal function
* [PR 189](https://github.com/pytroll/pyresample/pull/189) - Fix dask race condition in KDTree resampling

#### Features added

* [PR 183](https://github.com/pytroll/pyresample/pull/183) - Fix bb computation to generate areas with equal h and v resolutions

In this release 3 pull requests were closed.


## Version 1.12.1 (2019/04/24)

### Pull Requests Merged

#### Bugs fixed

* [PR 186](https://github.com/pytroll/pyresample/pull/186) - Fix support for pyproj-2 EPSG syntax

#### Documentation changes

* [PR 185](https://github.com/pytroll/pyresample/pull/185) - Fix argument order in get_area_def doc

In this release 2 pull requests were closed.


## Version 1.12.0 (2019/04/06)

### Issues Closed

* [Issue 178](https://github.com/pytroll/pyresample/issues/178) - Can't install pyresample on OSX Mojave

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 177](https://github.com/pytroll/pyresample/pull/177) - Fix dynamic omerc parameter computation to survive nans

#### Features added

* [PR 182](https://github.com/pytroll/pyresample/pull/182) - Implement striding and aggregation for Swath- and AreaDefinitions
* [PR 180](https://github.com/pytroll/pyresample/pull/180) - Remove radians from create_area_def and allow compatibility with pyproj-2.0+

In this release 3 pull requests were closed.


## Version 1.11.2 (2019/03/18)

### Pull Requests Merged

#### Documentation changes

* [PR 176](https://github.com/pytroll/pyresample/pull/176) - Fix typos in README and index page

In this release 1 pull request was closed.

## Version 1.11.1 (2019/03/18)

### Issues Closed

* [Issue 165](https://github.com/pytroll/pyresample/issues/165) - Update use of dask `atop` to `blockwise`
* [Issue 172](https://github.com/pytroll/pyresample/issues/172) - Missing metadata on PyPI ([PR 173](https://github.com/pytroll/pyresample/pull/173))

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 175](https://github.com/pytroll/pyresample/pull/175) - Fix dask.blockwise backwards compatibility with dask.atop

In this release 1 pull request was closed.


## Version 1.11.0 (2019/03/15)

### Issues Closed

* [Issue 160](https://github.com/pytroll/pyresample/issues/160) - No coastlines in cartopy if area is flipped ([PR 161](https://github.com/pytroll/pyresample/pull/161))
* [Issue 136](https://github.com/pytroll/pyresample/issues/136) - Update documentation to not reference scipy kdtree ([PR 155](https://github.com/pytroll/pyresample/pull/155))

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 170](https://github.com/pytroll/pyresample/pull/170) - Allow create_area_def to work with incomplete proj dicts to create DynamicAreas
* [PR 167](https://github.com/pytroll/pyresample/pull/167) - Fix yaml area loading when projection is latlong (units degrees)

#### Features added

* [PR 164](https://github.com/pytroll/pyresample/pull/164) - Delete unused requirements.txt file
* [PR 156](https://github.com/pytroll/pyresample/pull/156) - Refactor pyresample.utils in to separate modules
* [PR 150](https://github.com/pytroll/pyresample/pull/150) - Switch to versioneer
* [PR 145](https://github.com/pytroll/pyresample/pull/145) - Refactor and deprecate geometry "*_dask" methods
* [PR 138](https://github.com/pytroll/pyresample/pull/138) - Add `create_area_def` utility method and refactor AreaDefinition arguments

#### Documentation changes

* [PR 155](https://github.com/pytroll/pyresample/pull/155) - Update installation instructions to match current best practices ([136](https://github.com/pytroll/pyresample/issues/136))

In this release 8 pull requests were closed.


## Version 1.10.3 (2018/11/23)

### Issues Closed

* [Issue 92](https://github.com/pytroll/pyresample/issues/92) - Add utility function for converting geotiffs to area definitions ([PR 143](https://github.com/pytroll/pyresample/pull/143))

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 147](https://github.com/pytroll/pyresample/pull/147) - Fix dtype preservation for kdtree resampling
* [PR 144](https://github.com/pytroll/pyresample/pull/144) - Non-contiguous area definitions are now not concatenable ([491](https://github.com/pytroll/satpy/issues/491))

#### Features added

* [PR 143](https://github.com/pytroll/pyresample/pull/143) - get_area_def_from_raster ([92](https://github.com/pytroll/pyresample/issues/92))
* [PR 142](https://github.com/pytroll/pyresample/pull/142) - Add converter from def to yaml

In this release 4 pull requests were closed.


## Version 1.10.2 (2018/10/01)

### Issues Closed

* [Issue 133](https://github.com/pytroll/pyresample/issues/133) - Build issue with Python 3.7 ([PR 135](https://github.com/pytroll/pyresample/pull/135))

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 139](https://github.com/pytroll/pyresample/pull/139) - Fix area parsing code to convert PROJ.4 parameters to float if possible
* [PR 135](https://github.com/pytroll/pyresample/pull/135) - Fix Python 3.7 compatibility ([133](https://github.com/pytroll/pyresample/issues/133))

In this release 2 pull requests were closed.


## Version 1.10.1 (2018/07/03)

### Pull Requests Merged

#### Bugs fixed

* [PR 130](https://github.com/pytroll/pyresample/pull/130) - Fix log message not to rely on the proj key
* [PR 129](https://github.com/pytroll/pyresample/pull/129) - Ignore proj dicts with no  key for slicing

In this release 2 pull requests were closed.


## Version 1.10.0 (2018/06/25)

### Pull Requests Merged

#### Features added

* [PR 128](https://github.com/pytroll/pyresample/pull/128) - Add option to provide KDTree's 'mask' argument when querying

In this release 1 pull request was closed.


## Version 1.9.3 (2018/06/08)

### Issues Closed

* [Issue 113](https://github.com/pytroll/pyresample/issues/113) - Not all the close neighbours are found until search radius is increased ([PR 112](https://github.com/pytroll/pyresample/pull/112))
* [Issue 111](https://github.com/pytroll/pyresample/issues/111) - Bilinear interpolation leaves holes in fields with constant value ([PR 112](https://github.com/pytroll/pyresample/pull/112))

In this release 3 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 125](https://github.com/pytroll/pyresample/pull/125) - Fix area slices not working for non-geos projections
* [PR 119](https://github.com/pytroll/pyresample/pull/119) - Add hashing to StackedAreaDefinitions

In this release 1 pull request was closed.


## Version 1.9.2 (2018/05/13)

### Pull Requests Merged

#### Bugs fixed

* [PR 117](https://github.com/pytroll/pyresample/pull/117) - Fix get_area_slices ([218](https://github.com/pytroll/satpy/issues/218))

#### Features added

* [PR 116](https://github.com/pytroll/pyresample/pull/116) - Simplify get_sample_from_neighbour_info method

In this release 2 pull requests were closed.


## Version 1.9.1 (2018/05/03)

### Pull Requests Merged

#### Features added

* [PR 115](https://github.com/pytroll/pyresample/pull/115) - Geos area reduction

In this release 1 pull request was closed.
