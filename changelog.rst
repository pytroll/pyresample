Changelog
=========

%%version%% (unreleased)
------------------------

- Update changelog. [Martin Raspaud]

- Bump version: 1.1.5 → 1.1.6. [Martin Raspaud]

- Fix #35 supporting scipy kdtree again. [Martin Raspaud]

  A previous commit was looking for a 'data_pts' attribute in the kdtree
  object, which is available in pykdtree, but not scipy.

- Merge pull request #32 from mitkin/master. [Martin Raspaud]

  [tests] Skip deprecation warnings in test_gauss_multi_uncert

- Merge remote-tracking branch 'gh-pytroll/pre-master' [Mikhail Itkin]

- Put quotes around pip version specifiers to make things work. [Martin
  Raspaud]

- Install the right matplotlib in travis. [Martin Raspaud]

  The latest matplotlib (1.5) doesn't support python 2.6 and 3.3. This patch
  chooses the right matplotlib version to install depending on the python
  version at hand.

- Skip deprecation warnings. [Mikhail Itkin]

  Catch the rest of the warnings. Check if there is only one, and
  whether it contains the relevant message ('possible more than 8
  neighbours found'). This patch is necessary for python 2.7.9 and newer


- Merge pull request #31 from bhawkins/fix-kdtree-dtype. [Martin
  Raspaud]

  Fix possible type mismatch with pykdtree.

- Add test to expose pykdtree TypeError exception. [Brian Hawkins]

- Fix possible type mismatch with pykdtree. [Brian Hawkins]

- Update changelog. [Martin Raspaud]

- Bump version: 1.1.4 → 1.1.5. [Martin Raspaud]

- Don't build on 3.2 anymore (because of coverage's lack of support for
  3.2). [Martin Raspaud]

- Fix build badge adress. [Martin Raspaud]

- Fix the unicode problem in python3. [Martin Raspaud]

- Update changelog. [Martin Raspaud]

- Bump version: 1.1.3 → 1.1.4. [Martin Raspaud]

- Bugfix: Accept unicode proj4 strings. Fixes #24. [Martin Raspaud]

- Add python-configobj as a rpm requirement in setup.cfg. [Martin
  Raspaud]

- Add setup.cfg to allow rpm generation with bdist_rpm. [Martin Raspaud]

- Bugfix to address a numpy DeprecationWarning. [Martin Raspaud]

  Numpy won't take non-integer indices soon, so make index an int.

1.1.3 (2015-02-03)
------------------

- Merge branch 'release-1.1.3' [Martin Raspaud]

- Merge branch 'licence-lgpl' into pre-master. [Martin Raspaud]

- Switch to lgplv3, and bump up version number. [Martin Raspaud]

- Swith badge to main repository. [Martin Raspaud]

- Merge branch 'hotfix-v1.1.2' into pre-master. [Martin Raspaud]

1.1.2 (2014-12-17)
------------------

- Merge branch 'hotfix-v1.1.2' [Martin Raspaud]

- Bump up version number. [Martin Raspaud]

- Merge branch 'mitkin-master' into hotfix-v1.1.2. [Martin Raspaud]

- Merge branch 'master' of https://github.com/mitkin/pyresample into
  mitkin-master. [Martin Raspaud]

- [test_plot] allow travis to test plot.py. [Mikhail Itkin]

- [pip+travis] use `requirements.txt` [Mikhail Itkin]

  Use `requirements.txt` instead of setuptools' `extras_require`
  for installing basemap.

  That is because PyPi basemap version won't find libgeos library
  so we resolve to use latest basemap from git. `Extras_require` don't
  allow providing custom links, only PyPi package names, so we have to
  specify links in requirements.txt. `dependency_links` argument to
  `setup` call is meant for cruicial dependencies, not custom ones, so we
  don't use them neither.


- [README] markdown + build status. [Mikhail Itkin]

   * Using markdown extension, added `README` symlink
   * Added travis build status badge


- Remove pip `-e` switch. [Mikhail Itkin]

- Merge branch 'master' of github.com:mitkin/pyresample. [Mikhail Itkin]

- Don't use setup.py for basemap installation. [Mikhail Itkin]

  Instead of putting basemap and matplotlib into `extras_require`
  install them directly

- Don't use setup.py for basemap installation. [Mikhail Itkin]

  Instead of putting basemap and matplotlib into `extras_require`
  install them directly


- Using ubuntu GIS custom ppa. [Mikhail Itkin]

  Added custom ppa with more up-to-date libgeos dependencies

- Install extra requirements using pip functionality. [Mikhail Itkin]

- Added more meaningful "quicklooks" name. [Mikhail Itkin]

  Using quicklooks name as it's what matplotlib and basemap are needed for

- [setup] added plotting dependencies. [Mikhail Itkin]

  pyresample/plot requires two extra dependencies:
   * matplotlib
   * basemap


- [travis] added system dependencies. [Mikhail Itkin]

   * matplotlib requires libfreetype6-dev
   * basemap requires libgeos libgeos-c1 and libgeos-dev


- Merge branch 'release-v1.1.1' [Martin Raspaud]

- Restore API functionality by importing necessary modules in __init__
  [Martin Raspaud]

- Merge branch 'release-v1.1.1' into pre-master. [Martin Raspaud]

  Conflicts:
  	pyresample/geometry.py
  	pyresample/kd_tree.py
  	test/test_geometry.py


- Removing old test directory. [Martin Raspaud]

- Merge the hotfix and the unittest restructuring into the release
  branch. [Martin Raspaud]

- Merge branch 'release-v1.1.1' into hotfix-1.1.1. [Thomas Lavergne]

  Conflicts:
  	pyresample/geometry.py
  	test/test_geometry.py
  	test/test_grid.py


- Be specific about the valid range of longitudes. [Thomas Lavergne]

- Be more specific about the valid longitude range [-180:+180[. Add a
  test for utils.wrap_longitudes() [Thomas Lavergne]

- Add check on valid latitude in [-90:+90] (and associated test) [Thomas
  Lavergne]

- Automatic longitude wrapping (bugfix towards 1.1.1) [Thomas Lavergne]

- Merge branch 'release-v1.1.1' into pre-master. [Martin Raspaud]

- Merge branch 'pre-master' of https://code.google.com/p/pyresample into
  pre-master. [Martin Raspaud]

- A stray line of code is removed and I take back the recent enhancement
  concerning swath to swath mapping. [Adam Dybbroe]

- Removed debug printouts. [Adam Dybbroe]

- More active support of swath to swath reprojection. [Adam Dybbroe]

- Add a plot on multiprocessing performance increases. [Martin Raspaud]

- Added outer_boundary_corners property to the area def class. [Adam
  Dybbroe]

1.1.1 (2014-12-10)
------------------

- Merge branch 'release-v1.1.1' [Martin Raspaud]

- Add news about new release. [Martin Raspaud]

- Remove some relative imports. [Martin Raspaud]

- Cleanup and bump up version number to v1.1.1. [Martin Raspaud]

- Add pykdtree to the list of requirements for travis. [Martin Raspaud]

- Add .travis.yml file for automatic testing. [Martin Raspaud]

- Correct handling of long type in kd_tree.py for Python 2. [Martin
  Valgur]

- Made testing of a Proj4 string independent of the order of elements
  inside the string since the order was different on Python 2 and 3.
  Replaced deprecated failIf with assertFalse. [Martin Valgur]

- Multiple small fixes to make the code work on both Python 2 and 3.
  shmem_as_ndarray() now uses numpy.frombuffer() to provide equivalent
  functionality. [Martin Valgur]

- Got rid of dependencies on the six package. [Martin Valgur]

- Applied python-modernize to pyresample. [Martin Valgur]

- Update README. [Martin Raspaud]

- Corrected docs. [Esben S. Nielsen]

- Modified uncert count to show above 0. Updated docs to relect uncert
  option. [Esben S. Nielsen]

- Cleaned up code a bit in kd_tree.py. [Esben S. Nielsen]

- Made API doc work with readthedocs and bumped version number. [Esben
  S. Nielsen]

- Cleaned up code and tests. [Esben S. Nielsen]

- Added masking of uncert counts. [Esben S. Nielsen]

- Test passes again for uncertainty calculations. [Esben S. Nielsen]

- Changed uncertainty API. First working uncertainty version. [Esben S.
  Nielsen]

- Not quite there. [Esben S. Nielsen]

- Basic uncertainty implemented. [Esben S. Nielsen]

- Updated docs. [Esben S. Nielsen]

- Fixing bug, and adding unittest-main run. [Adam Dybbroe]

- Making get_xy_from_lonlat work on arrays of points as well as single
  points. [Adam Dybbroe]

- Renamed functions in geometry.py and added proj_x_coords and
  proj_y_coords properties. [Esben S. Nielsen]

- Corrected __eq__ in geometry. [Esben S. Nielsen]

- Merge branch 'pre-master' of https://code.google.com/p/pyresample into
  pre-master. [Adam Dybbroe]

- Now kd_tree resampling selects dtype. [Esben S. Nielsen]

- Removed random print statement. [Esben S. Nielsen]

- Made get_capabilites function. [Esben S. Nielsen]

- Test passes again. [Esben S. Nielsen]

- Removed caching from geometry. [Esben S. Nielsen]

- Merge branch 'pre-master' of https://code.google.com/p/pyresample into
  pre-master. [Martin Raspaud]

- Optimize transform_lonlats with numexpr. [Martin Raspaud]

- Unittests should work for both py2.6 and 2.7. [Adam Dybbroe]

- Updated docs. [Esben S. Nielsen]

- Fixed unit tests. [Esben S. Nielsen]

- Using assertRaises in py2.6 and py2.7 compatible version. [Adam
  Dybbroe]

- Bugfix to unittest suite. [Adam Dybbroe]

- Trying to make test-functions compatible with both python 2.6 and 2.7.
  [Adam Dybbroe]

- Fixing bug in get_xy_from_lonlat and adding unittests on this
  function. [Adam Dybbroe]

- Adding function get_xy_from_lonlat. [Adam Dybbroe]

- Integrated pykdtree and handled latlong projection bug. [Esben S.
  Nielsen]

- Updated unit tests according to deprecation warnings. [Esben S.
  Nielsen]

- Better parsing of a area definition (allow ':' in value fields) [Lars
  Orum Rasmussen]

- Updated docs. [Esben S. Nielsen]

- Merge branch 'pre-master' of https://code.google.com/p/pyresample into
  pre-master. [Martin Raspaud]

- Doc version. [esn]

- Improved Basemap integration with globe projections. Updated docs on
  epsilon. [esn]

- Accomodate for allclose behaviour change in numpy 1.6.2. [Martin
  Raspaud]

  From 1.6.2 numpy.allclose does not accept arrays that cannot be
  broadcasted to the same shape. Hence a ValueError catch to return False.


- Updadet doc for plotting. [Esben S. Nielsen]

- Updated plot test to use AGG. [Esben S. Nielsen]

- Now handles plotting in Plate Carre projection. Added utils.fwhm2sigma
  function. [Esben S. Nielsen]

- Merge branch 'master' of https://code.google.com/p/pyresample. [Esben
  S. Nielsen]

- Added pypi info. [Esben S. Nielsen]

- Built docs. [Esben S. Nielsen]

- Corrected test_swath.py to account for implementation specific
  precision. [Esben S. Nielsen]

- More datatype specifications. [Esben S. Nielsen]

- Removed warning check for python 2.5. [Esben S. Nielsen]

- Corrected multi channnel bug. Added warnings for potential problematic
  neighbour query condition. [Esben S. Nielsen]

- Now str() generates a unique string for area and coordinate definition
  object. [Lars Orum Rasmussen]

- Corrected manifest so doc images are included. [Esben S. Nielsen]

- Moved tests dir to test. Updated MANIFEST.in. [Esben S. Nielsen]

- Added MANIFEST.in. [Esben S. Nielsen]

- Applied setup.py patches. Made plotting more robust. [Esben S.
  Nielsen]

- Applied patch for getting version number. [Esben S. Nielsen]

- Bugfixing quicklooks. [StorPipfugl]

- Updated docs. [StorPipfugl]

- Updated docs. [StorPipfugl]

- Updated docs. [StorPipfugl]

- Added Basemap integration. [StorPipfugl]

- Added Basemap integration. [StorPipfugl]

- Updated docs. [StorPipfugl]

- Rebuild docs. [StorPipfugl]

- Made setup.py more robust. [StorPipfugl]

- New doc version. [StorPipfugl]

- Updated tests. [StorPipfugl]

- Reduced size of linesample arrays. Restructures kd_tree query to
  remove redundant lon lat calculations. [StorPipfugl]

- Added geographic filtering. Swaths can now be concatenated and
  appended. User no langer have to ravel data before resampling.
  [StorPipfugl]

- Updated docs. [StorPipfugl]

- Updated install_requires. [StorPipfugl]

- Version 0.7.3. [StorPipfugl]

- Bugfixes: Correct number of channels in empty result set. Resampling
  of masked data to 1d swath now works. [StorPipfugl]

- Added Martin's spherical geometry operations. Updated documentation.
  [StorPipfugl]

- Added equal and not equal operators for geometry defs. Restructured
  the geometry module to be pickable. Added correct handling of empty
  result data sets. [StorPipfugl]

- Incomplete - taskpyresample. [StorPipfugl]

- Set svn:mime-type. [StorPipfugl]

- Corrected doc errors. [StorPipfugl]

- Removed dist dir. [StorPipfugl]

- Updated documentation. New release. [StorPipfugl]

- Started updating docstrings. [StorPipfugl]

- Restructured API. [StorPipfugl]

- Now uses geometry types. Introduced API symmetry between swath->grid
  and grid->swath resampling. [StorPipfugl]

- Consolidated version tag. [StorPipfugl]

- Mime types set. [StorPipfugl]

- Mime types set. [StorPipfugl]

- Removed test. [StorPipfugl]

- Removed unneeded function. [StorPipfugl]

- Mime types set. [StorPipfugl]

- Mime types set. [StorPipfugl]

- Moved to Google Code under GPLv3 license. [StorPipfugl]

- Moved to Google Code. [StorPipfugl]


