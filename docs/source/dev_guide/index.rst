Developer's Guide
=================

The below sections and documents describe information for people contributing
to Pyresample.

Writing documentation
---------------------

The pyresample sphinx documentation is structured following the information
described in this presentation by Daniele Procida:
https://youtu.be/t4vKPhjcMZg

For more details on this scheme see:

https://documentation.divio.com/

The documentation is split into 4 main groups:

1. Concepts (topics): The high-level concepts involved with using pyresample.
   This section tries to not include any code examples and sticks to pure text
   descriptions of the topics.
2. Tutorials: Simple end-to-end examples with provided data (fake or real) that
   walk users through a series of pyresample steps to accomplish an overall
   goal. Tutorials should avoid long detailed explanations. Tutorials assume
   the reader doesn't have enough knowledge to know what they want to
   accomplish exactly and the tutorial should show them what's possible or
   rather what can be done with pyresample.
3. How-Tos: Short and exact examples showing how a single feature or small
   collection of features are used.
   How-tos can assume the user is more familiar with concepts and provide a
   little more detail. They can also assume the user has their own data and use
   case for wanting to perform the operations described in the how-to.
   Put another way, the user knows what they want to do, the how-to shows them
   how to do it. How-tos should still not go into any more detail than
   necessary and try to depend on :doc:`../concepts/index`.
4. Reference: API docs and other low-level information.
