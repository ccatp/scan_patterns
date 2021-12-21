# mapping

[![Documentation Status](https://readthedocs.org/projects/mapping-fyst/badge/?version=latest)](https://mapping-fyst.readthedocs.io/en/latest/?badge=latest)

``mapping`` is a Python software package for simulating on-sky data acquisition and forecasting scanning performance. 
The ``mapping`` package allows for tools that:

* Configuring camera instruments with user-defined and/or pre-defined detector arrays. 
* Representing different scan patterns in terms of its offsets as well as its telescope motion. 
* Determining optimal times for observing a particular source.

While default options are specific to PrimeCam and the Fred Young Submillimetre Telescope (FYST),
the package can also be used for other telescopes and cameras. 

See the full documentation here: 

https://mapping-fyst.readthedocs.io/en/latest/

To interact with the [tutorial notebooks](tutorials) without installation, you can use this:

https://mybinder.org/v2/gh/KristinChengWu/mapping/HEAD

##  Installation

For local installations, you can install the latest version

```
pip install git+https://github.com/KristinChengWu/mapping.git#egg=mapping
```

or upgrade an existing version. 

```
pip install -U git+https://github.com/KristinChengWu/mapping.git#egg=mapping
```

For use in Docker or other self-contained python projects use the following line in the requirements.txt file:

```
-e git+https://github.com/KristinChengWu/mapping.git#egg=mapping
```