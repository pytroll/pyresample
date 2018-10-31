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
