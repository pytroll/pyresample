# Documentation Creation

With sphinx and pyresample's dependencies installed documentation can be generated
by running:

    make html

The generated HTML documentation pages are available in `build/html`. If
Pyresample's API has changed (new functions, modules, classes, etc) then the
API documentation should be regenerated before running the above make
command.

    sphinx-apidoc -f -T -o source/api ../pyresample ../pyresample/test
