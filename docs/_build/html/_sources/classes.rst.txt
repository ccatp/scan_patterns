######################
Classes
######################

A reference for the documentation of all user class objects. 

Module
=============

.. autoclass:: scanning.camera.Module
    :special-members: __init__
    :members:

Instrument
==============

.. autoclass:: scanning.camera.Instrument
    :special-members: __init__
    :members:

PrimeCam
------------

A subclass of the ``Instrument`` class. The only major difference are the slot locations, so inherited members are not listed.

.. autoclass:: scanning.camera.PrimeCam
    :members:

ModCam
------------

A subclass of the ``Instrument`` class. The only major difference are the slot locations, so inherited members are not listed.

.. autoclass:: scanning.camera.ModCam
    :members:

SkyPattern
====================

.. autoclass:: scanning.coordinates.SkyPattern
    :special-members: __init__
    :members:

Pong
-----------

A subclass of the ``SkyPattern`` object. Inherited members are listed. 

.. autoclass:: scanning.coordinates.Pong
    :special-members: __init__
    :members:
    :inherited-members:

Daisy
------------

A subclass of the ``SkyPattern`` object. Inherited members are listed. 

.. autoclass:: scanning.coordinates.Daisy
    :special-members: __init__
    :members:
    :inherited-members:

TelescopePattern
=========================

.. autoclass:: scanning.coordinates.TelescopePattern
    :special-members: __init__
    :members:

Simulation
=================

.. autoclass:: scanning.optimization.Simulation
    :special-members: __init__
    :members:

Observation
===================

.. autoclass:: scanning.observation.Observation
    :special-members: __init__
    :members:
