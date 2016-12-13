Changelog
=========

v1.2.9 (2016-12-13)
-------------------

- update changelog. [Martin Raspaud]

- Bump version: 1.2.8 → 1.2.9. [Martin Raspaud]

- Merge pull request #52 from mitkin/mitkin-pr-setuptools32. [Martin
  Raspaud]

  Specify minimum version of setuptools

- Specify minimum version of setuptools. [Mikhail Itkin]

  Prior to version 3.2 setuptools would not recognize correctly the language of `*.cpp` extensions and would assume it's `*.c` no matter what. Version 3.2 of setuptools fixes that.

- Fix sphinx dependency to support python 2.6 and 3.3. [Martin Raspaud]

v1.2.8 (2016-12-06)
-------------------

- update changelog. [Martin Raspaud]

- Bump version: 1.2.7 → 1.2.8. [Martin Raspaud]

- Correct style in setup.py. [Martin Raspaud]

- Make pykdtree a requirement. [Martin Raspaud]

- Correct style in geometry.py. [Martin Raspaud]

- Allow precision errors when comparing area_extents. [Martin Raspaud]

- Allow numbers in proj dict when building proj4 string. [Martin
  Raspaud]

v1.2.7 (2016-11-15)
-------------------

- update changelog. [Martin Raspaud]

- Bump version: 1.2.6 → 1.2.7. [Martin Raspaud]

- Add bump and changelog config files. [Martin Raspaud]

- Merge pull request #49 from Funkensieper/fix-polygon-area. [Martin
  Raspaud]

  Fix polygon area

- Disable snapping of angles in get_polygon_area() [Stephan
  Finkensieper]

  - Add option to disable snapping in Arc.angle()
  - Don't snap angles when computing polygon areas in order to
    prevent negative area values
  - Adjust reference values in tests on overlap-rate


- Fix polygon area computation for R != 1. [Stephan Finkensieper]

  Parentheses were missing, see

  http://mathworld.wolfram.com/SphericalTriangle.html

  for reference. Only affects earth radius R != 1 which is not
  implemented yet.


- Install pykdtree from conda forge in pre-master. [davidh-ssec]

- Merge pull request #47 from mitkin/feature_plot-cmap. [David Hoese]

  Add option to choose colormap

- Add option to choose colormap. [Mikhail Itkin]

  Make possible to indicate which colormap to use when plotting image


v1.2.6 (2016-10-19)
-------------------

- update changelog. [Martin Raspaud]

- Bump version: 1.2.5 → 1.2.6. [Martin Raspaud]

- Pre master (#44) [Radar, Satellite and Nowcasting Division]

  * add a subset function to the geometry file

  Often subsets of the SEVIRI disk are save in
  SEVIRI products. This new function calculated the
  area extent needed for the specification of the area,
  starting from the SEVIRI full disk area object.

  * added the get_area_extent_for_subset function

  to the geometry.py file

  * new function for getting area subsets

  * new function get_xy_from_proj_coordinates

  retrieve the pixel indices x and y
  from the map projection coordinates in meter
  (very similar to get_xy_from_lonlat)

  * removed pyc file, that should not be in the git repository


- Add appveyor status badge to README. [davidh-ssec]

- Merge remote-tracking branch 'deni90/master' into pre-master-davidh.
  [davidh-ssec]

- Fix test_custom_uncert and test_gauss_uncert for mips* [Daniel
  Knezevic]

- Fix pykdtree install on appveyor by turning off OpenMP. [davidh-ssec]

- Update appveyor config to install missing headers required by
  pykdtree. [davidh-ssec]

- Change appveyor to use conda-forge instead of IOOS. [davidh-ssec]

- Add slack notifications from appveyor. [davidh-ssec]

v1.2.5 (2016-07-21)
-------------------

- update changelog. [Martin Raspaud]

- Bump version: 1.2.4 → 1.2.5. [Martin Raspaud]

- Fix EWA fornav for integer data and add test. [davidh-ssec]

  There was a bug when doing the averaging in EWA where the current pixel was being rounded based on the previous pixel's value instead of the current pixel. This only affects integer data because values above 0 are rounded up by 0.5 and values below 0 are rounded by 0.5, for floats this round value is 0.0.


- Fix certain compilers not liking integers being passed to isnan.
  [davidh-ssec]

- Replace catch_warnings in all tests with astropy version. [davidh-
  ssec]

- Use catch_warnings from astropy (or at least try to) [davidh-ssec]

- Test removing version specific warning checks in `test_swath_wrap`
  [davidh-ssec]

- Move USE_CYTHON handling to if main block in setup.py. [davidh-ssec]

- Fix isnan definition only if a macro doesn't already exist. [davidh-
  ssec]

  Numpy does some special macro stuff to define a good npy_isnan function. Some systems define a macro for it, others don't. Hopefully this works for all systems. A better solution might be to define a templated isnan that calls npy_isnan if it isn't an integer.


- fix EWA compile failure on windows python 3.5. [David Hoese]

- Make pykdtree install on appveyor optional. [davidh-ssec]

- Add pykdtree to appveyor dependencies. [davidh-ssec]

- Fix setup.py test on windows for multiprocessing tests. [davidh-ssec]

  On Windows when new processes are started the initially command is imported or re-executed. For setup.py this is a big problem since the usual boilerplate does not include `if __name__ == "__main__"` so the setup.py test command gets rerun and rerun. This results in the child processes never actually being run for newer versions of python (2.7+). There still seems to be an issue with `test_nearest_resize` on Windows.


- Merge pull request #41 from cpaulik/fix-windows-ewa. [David Hoese]

  Fix Windows CI import Error

- Install scipy in Windows CI to fix import problems. [Christoph Paulik]

- Fix copy/paste error in EWA fornav. [davidh-ssec]

  I had started rewriting EWA in cython then realized it was faster in straight C++ so copied/pasted the cython code and modified it. Seems like I missed this 'or' hanging around.


- Fix NAN constant/macro for EWA on Windows. [davidh-ssec]

- Merge branch 'add-windows-CI' into fix-windows-ewa. [davidh-ssec]

- CI: Add IOOS conda channel to get basemap for Windows and python > 2.
  [Christoph Paulik]

- Merge branch 'add-windows-CI' into fix-windows-ewa. [davidh-ssec]

- Add pyproj to conda install in Appveyor CI. [Christoph Paulik]

- Make extra_compile_args platform dependent. [Christoph Paulik]

- Add Appveyor CI configuration. [Christoph Paulik]

- Fix EWA resampling's isnan to work better with windows. [davidh-ssec]

v1.2.4 (2016-06-27)
-------------------

- update changelog. [Martin Raspaud]

- Bump version: 1.2.3 → 1.2.4. [Martin Raspaud]

- Fix setup.py extension import and use error. [davidh-ssec]

- Fix case when __builtins__ is a dict. [Martin Raspaud]

v1.2.3 (2016-06-21)
-------------------

- update changelog. [Martin Raspaud]

- Bump version: 1.2.2 → 1.2.3. [Martin Raspaud]

- Fix list of package names in setup.py. [davidh-ssec]

  'pyresample.ewa' wasn't listed before and was not importable from an installed package.


v1.2.2 (2016-06-21)
-------------------

- update changelog. [Martin Raspaud]

- Bump version: 1.2.1 → 1.2.2. [Martin Raspaud]

- Add the header files to the MANIFEST.in. [Martin Raspaud]

  Without this, the compilation of the ewa extension crashes.

v1.2.1 (2016-06-21)
-------------------

- update changelog. [Martin Raspaud]

- Bump version: 1.2.0 → 1.2.1. [Martin Raspaud]

- Include EWA header files as dependency for extensions. [davidh-ssec]

  The .c and .cpp files are automatically included because they are listed as sources, but the header files are not. When building a source tarball (uploading to PyPI) the _fornav_templates.h file was not included and building would fail.


- Merge branch 'pre-master' of github.com:mraspaud/pyresample into pre-
  master. [Adam.Dybbroe]

- Merge branch 'pre-master' of github.com:mraspaud/pyresample into pre-
  master. [Adam.Dybbroe]

  Conflicts:
  	docs/source/conf.py


- Run the base class init function first. [Adam.Dybbroe]

v1.2.0 (2016-06-17)
-------------------

- update changelog. [Martin Raspaud]

- Bump version: 1.1.6 → 1.2.0. [Martin Raspaud]

- Merge branch 'northaholic-feature-lonlat2colrow' into pre-master.
  [Adam.Dybbroe]

- Add two convenience methods lonlat2colrow and colrow2lonlat to
  AreaDefinition-class. [Sauli Joro]

- Fix bug in EWA grid origin calculation. [davidh-ssec]

  Forgot that cell height was negative so ended up subtracting a negative, going in the wrong direction for the Y origin of the grid.


- Merge pull request #37 from davidh-ssec/feature-ewa-resampling. [David
  Hoese]

  Feature ewa resampling

- Fix bug in EWA conversion from AreaDefinition to upper-left origin
  X/Y. [davidh-ssec]

  I was using the area extent for the origin x/y locations, but the extent is actually the outer edge of the pixels so half a pixel needs to be added to each coordinate.


- Add EWA C extensions to mocked modules for read the docs. [davidh-
  ssec]

  Readthedocs.org fails to import the _ll2cr and _fornav extensions because it seems to not compile them properly. Their documentation isn't necessarily needed so I'm hoping that mocking them will let the import work.


- Add pyresample.ewa to API documentation list. [davidh-ssec]

- Update EWA wrapper functions to use explicit kwargs. [davidh-ssec]

- Correct comments and documentation in EWA documentation. [davidh-ssec]

- Add ll2cr and fornav wrappers to make calling easier. [davidh-ssec]

  Updated documentation with correct usage and added information why EWA is different than kdtree


- Fix print statements in documentation so doctests are python 3
  compatible. [davidh-ssec]

- Add pillow dependency for plot tests and quicklook extra. [davidh-
  ssec]

- Add 'areas.cfg' file to repository and modify doctests to use that
  instead. [davidh-ssec]

- Run doctests after unittests on travis. [davidh-ssec]

- Fix documentation for AreaDefinition object. [davidh-ssec]

- Update documentation to be numpy style and get rid of all warnings
  when building docs. [davidh-ssec]

- Create special requirements.txt for docs. [davidh-ssec]

  Readthedocs really doesn't like an empty string for the requirements file


- Try empty string for requirements file in readthedocs yaml. [davidh-
  ssec]

- Fix readthedocs yaml config file. [davidh-ssec]

  Readthedocs was using the requirements file during package installation, but was failing to install basemap (not needed for documentation build) so I attempted to make it an empty string in the yaml file. This makes Rtd hang on the build process. This should at least stop the hanging.


- Add napoleon docs extension and intial testing with numpy style
  docstrings. [davidh-ssec]

- Add working example for EWA resampling to docs. [davidh-ssec]

  I originally had this example but removed it when I had import problems. After I figured those out I forgot to put the original example back.


- Add basemap back in to the requirements.txt so that it can be
  installed on travis. [davidh-ssec]

  Similarly removed the requirements file when readthedocs is running and mocked third-party packages to documentation can still be built


- Fix setup.py requiring numpy for extension includes. [davidh-ssec]

  The EWA extensions require the numpy headers to be built. These are normally found by importing numpy and doing `numpy.get_includes()`. Obviously if this is run on a new environment numpy is probably not installed so a simple `python setup.py install` will fail.


- Add "quicklook" extra in to travis test install. [davidh-ssec]

  These packages are needed to properly test the "plot" package. These were included in requirements.txt but have been moved for now.


- Move plot test imports in to test functions for cleaner test failures.
  [davidh-ssec]

- Add readthedocs yaml file for configuration. [davidh-ssec]

- Remove mocked modules from sphinx docs conf.py. [davidh-ssec]

  This is the first step in making pyresamples docs buildable in the current readthedocs version


- Replace relative imports with absolute imports. [davidh-ssec]

  I noticed a lot of warnings and import problems with building pyresample's documentation because of these relative imports


- Add EWA documentation to swath.rst. [davidh-ssec]

- Add tests for EWA fornav module. [davidh-ssec]

- Update documentation for ll2cr and fornav cython. [davidh-ssec]

- Merge remote-tracking branch 'davidh_fork/feature-ewa-resampling' into
  feature-ewa-resampling. [davidh-ssec]

  # Conflicts:
  #	pyresample/ewa/_fornav.pyx
  #	pyresample/ewa/_ll2cr.pyx


- Remove old and unused polar2grid ll2cr and fornav python modules.
  [davidh-ssec]

- Fix travis tests on python 2.6. [davidh-ssec]

- Add ewa ll2cr tests to main test suite. [davidh-ssec]

- Add simple tests for ewa ll2cr. [davidh-ssec]

  These tests were adapted from Polar2Grid so some of the terminology or organization might reflect P2G's design rather than satpy or pyresample.


- Revert import multiprocessing setup.py for python 2.6 compatibility.
  [davidh-ssec]

- Fix old polar2grid import in ll2cr module. [davidh-ssec]

- Add method for converting area def to areas.def string format.
  [davidh-ssec]

- Remove unused code from fornav wrapper. [davidh-ssec]

- Add initial EWA files copied from Polar2Grid. [davidh-ssec]

- Add basic documentation to fornav cython function. [davidh-ssec]

- Remove old and unused polar2grid ll2cr and fornav python modules.
  [davidh-ssec]

- Fix travis tests on python 2.6. [davidh-ssec]

- Add ewa ll2cr tests to main test suite. [davidh-ssec]

- Add simple tests for ewa ll2cr. [davidh-ssec]

  These tests were adapted from Polar2Grid so some of the terminology or organization might reflect P2G's design rather than satpy or pyresample.


- Revert import multiprocessing setup.py for python 2.6 compatibility.
  [davidh-ssec]

- Fix old polar2grid import in ll2cr module. [davidh-ssec]

- Add method for converting area def to areas.def string format.
  [davidh-ssec]

- Remove unused code from fornav wrapper. [davidh-ssec]

- Add initial EWA files copied from Polar2Grid. [davidh-ssec]

- Add .gitignore with python and C patterns. [davidh-ssec]

- Update tests so they don't fail on OSX. [davidh-ssec]

  OSX seems to calculate slightly different results from `_spatial_mp.Cartesian` regardless of numexpr being installed. Although the changes are small they seem to affect the results enough to fail this test compared to normal linux execution.


- Add 'load_tests' for easier test selection. [davidh-ssec]

  PyCharm and possibly other IDEs don't really play well with unittest TestSuites, but work as expected when `load_tests` is used.


- Make kd_tree test work on older numpy version. [Martin Raspaud]

  VisibleDeprecationWarning is not available in numpy <1.9.

- Adapt to newest pykdtree version. [Martin Raspaud]

  The kdtree object's attribute `data_pts` has been renamed to `data`.

- Run tests on python 3.5 in travis also. [Martin Raspaud]

v1.1.6 (2016-02-25)
-------------------

- update changelog. [Martin Raspaud]

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

v1.1.5 (2015-10-12)
-------------------

- update changelog. [Martin Raspaud]

- Bump version: 1.1.4 → 1.1.5. [Martin Raspaud]

- Don't build on 3.2 anymore (because of coverage's lack of support for
  3.2). [Martin Raspaud]

- Fix build badge adress. [Martin Raspaud]

- Fix the unicode problem in python3. [Martin Raspaud]

v1.1.4 (2015-10-08)
-------------------

Fix
~~~

- Bugfix: Accept unicode proj4 strings. Fixes #24. [Martin Raspaud]

Other
~~~~~

- update changelog. [Martin Raspaud]

- Bump version: 1.1.3 → 1.1.4. [Martin Raspaud]

- Add python-configobj as a rpm requirement in setup.cfg. [Martin
  Raspaud]

- Add setup.cfg to allow rpm generation with bdist_rpm. [Martin Raspaud]

- Bugfix to address a numpy DeprecationWarning. [Martin Raspaud]

  Numpy won't take non-integer indices soon, so make index an int.

- Merge branch 'release-1.1.3' [Martin Raspaud]

- Merge branch 'licence-lgpl' into pre-master. [Martin Raspaud]

- Switch to lgplv3, and bump up version number. [Martin Raspaud]

- Swith badge to main repository. [Martin Raspaud]

- Merge branch 'hotfix-v1.1.2' into pre-master. [Martin Raspaud]

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


- remove pip `-e` switch. [Mikhail Itkin]

- Merge branch 'master' of github.com:mitkin/pyresample. [Mikhail Itkin]

- don't use setup.py for basemap installation. [Mikhail Itkin]

  Instead of putting basemap and matplotlib into `extras_require`
  install them directly

- don't use setup.py for basemap installation. [Mikhail Itkin]

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

- Add news about new release. [Martin Raspaud]

- remove some relative imports. [Martin Raspaud]

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

- Merge branch 'pre-master' of https://code.google.com/p/pyresample into
  pre-master. [Martin Raspaud]

- A stray line of code is removed and I take back the recent enhancement
  concerning swath to swath mapping. [Adam Dybbroe]

- Removed debug printouts. [Adam Dybbroe]

- More active support of swath to swath reprojection. [Adam Dybbroe]

- Add a plot on multiprocessing performance increases. [Martin Raspaud]

- Added outer_boundary_corners property to the area def class. [Adam
  Dybbroe]

- corrected docs. [Esben S. Nielsen]

- modified uncert count to show above 0. Updated docs to relect uncert
  option. [Esben S. Nielsen]

- cleaned up code a bit in kd_tree.py. [Esben S. Nielsen]

- made API doc work with readthedocs and bumped version number. [Esben
  S. Nielsen]

- cleaned up code and tests. [Esben S. Nielsen]

- added masking of uncert counts. [Esben S. Nielsen]

- test passes again for uncertainty calculations. [Esben S. Nielsen]

- changed uncertainty API. First working uncertainty version. [Esben S.
  Nielsen]

- not quite there. [Esben S. Nielsen]

- basic uncertainty implemented. [Esben S. Nielsen]

- updated docs. [Esben S. Nielsen]

- Fixing bug, and adding unittest-main run. [Adam Dybbroe]

- Making get_xy_from_lonlat work on arrays of points as well as single
  points. [Adam Dybbroe]

- renamed functions in geometry.py and added proj_x_coords and
  proj_y_coords properties. [Esben S. Nielsen]

- corrected __eq__ in geometry. [Esben S. Nielsen]

- Merge branch 'pre-master' of https://code.google.com/p/pyresample into
  pre-master. [Adam Dybbroe]

- now kd_tree resampling selects dtype. [Esben S. Nielsen]

- removed random print statement. [Esben S. Nielsen]

- made get_capabilites function. [Esben S. Nielsen]

- test passes again. [Esben S. Nielsen]

- removed caching from geometry. [Esben S. Nielsen]

- Merge branch 'pre-master' of https://code.google.com/p/pyresample into
  pre-master. [Martin Raspaud]

- Optimize transform_lonlats with numexpr. [Martin Raspaud]

- Unittests should work for both py2.6 and 2.7. [Adam Dybbroe]

- updated docs. [Esben S. Nielsen]

- fixed unit tests. [Esben S. Nielsen]

- Using assertRaises in py2.6 and py2.7 compatible version. [Adam
  Dybbroe]

- bugfix to unittest suite. [Adam Dybbroe]

- Trying to make test-functions compatible with both python 2.6 and 2.7.
  [Adam Dybbroe]

- Fixing bug in get_xy_from_lonlat and adding unittests on this
  function. [Adam Dybbroe]

- Adding function get_xy_from_lonlat. [Adam Dybbroe]

- integrated pykdtree and handled latlong projection bug. [Esben S.
  Nielsen]

- updated unit tests according to deprecation warnings. [Esben S.
  Nielsen]

- Better parsing of a area definition (allow ':' in value fields) [Lars
  Orum Rasmussen]

- updated docs. [Esben S. Nielsen]

- Merge branch 'pre-master' of https://code.google.com/p/pyresample into
  pre-master. [Martin Raspaud]

- doc version. [esn]

- improved Basemap integration with globe projections. Updated docs on
  epsilon. [esn]

- Accomodate for allclose behaviour change in numpy 1.6.2. [Martin
  Raspaud]

  From 1.6.2 numpy.allclose does not accept arrays that cannot be
  broadcasted to the same shape. Hence a ValueError catch to return False.


- updadet doc for plotting. [Esben S. Nielsen]

- updated plot test to use AGG. [Esben S. Nielsen]

- Now handles plotting in Plate Carre projection. Added utils.fwhm2sigma
  function. [Esben S. Nielsen]

- Merge branch 'master' of https://code.google.com/p/pyresample. [Esben
  S. Nielsen]

- added pypi info. [Esben S. Nielsen]

- built docs. [Esben S. Nielsen]

- corrected test_swath.py to account for implementation specific
  precision. [Esben S. Nielsen]

- more datatype specifications. [Esben S. Nielsen]

- removed warning check for python 2.5. [Esben S. Nielsen]

- corrected multi channnel bug. Added warnings for potential problematic
  neighbour query condition. [Esben S. Nielsen]

- Now str() generates a unique string for area and coordinate definition
  object. [Lars Orum Rasmussen]

- corrected manifest so doc images are included. [Esben S. Nielsen]

- Moved tests dir to test. Updated MANIFEST.in. [Esben S. Nielsen]

- Added MANIFEST.in. [Esben S. Nielsen]

- Applied setup.py patches. Made plotting more robust. [Esben S.
  Nielsen]

- applied patch for getting version number. [Esben S. Nielsen]

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

- version 0.7.3. [StorPipfugl]

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

- No commit message. [StorPipfugl]

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

- No commit message. [StorPipfugl]

- Moved to Google Code under GPLv3 license. [StorPipfugl]

- moved to Google Code. [StorPipfugl]


