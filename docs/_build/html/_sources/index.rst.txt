.. mapping documentation master file, created by
   sphinx-quickstart on Sun Dec 12 18:00:59 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#####################
Introduction
#####################

``mapping`` is a Python software package for simulating on-sky data acquisition and forecasting scanning performance. 
The ``mapping`` package allows for tools that:

* Configuring camera instruments with user-defined and/or pre-defined detector arrays. 
* Representing different scan patterns in terms of its offsets as well as its telescope motion. 
* Determining optimal times for observing a particular source.

While default options are specific to PrimeCam and the Fred Young Submillimetre Telescope (FYST),
the package can also be used for other telescopes and cameras. 

Contents
===============

.. toctree::
   :maxdepth: 2

   install.rst
   tutorial.rst
   classes.rst
   visualization.rst

Indices and tables
=====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
