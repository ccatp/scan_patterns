(not up to date)

# Installation

For local installations from the gitlab version, you can install the latest version:
```
pip install git+https://github.com/KristinChengWu/mapping.git#egg=mapping
```

Or upgrade an existing version (note the -U):
```
pip install -U git+https://github.com/KristinChengWu/mapping.git#egg=mapping
```

For use in Docker or other self-contained python projects use the following line in the requirements.txt file:
```
-e git+https://github.com/KristinChengWu/mapping.git#egg=mapping
```

# Usage

The `scanning` Python package contains functions and classes for generating and plotting scan patterns. 

You can specify a type of scan pattern (e.g. `CurvyPong`) and initialize it. By default, angle-like units are in degrees and time-like units are in seconds, but you can specify units in the arguments as well. 

Below is an example of a curvy pong scan using five terms in its Fourier expansion, 2 deg by 7000 arcsec FoV, spacing (between scan lines) of 500 arcsec, a target velocity of 1000 arcsec/second and read-out of 0.002 seconds. 

``` py
import astropy.units as u
from scanning.scan_patterns import CurvyPong

scan = CurvyPong(num_terms=5, width=2, height=7000*u.arcsec, spacing='500 arcsec', velocity='1000 arcsec/s', sample_interval=0.002)
```

For AZ/ALT related values, you can set the scan's setting like the following (RA of 0 degrees, declination of 0 degrees, altitude of 60 degrees). By default, location is at FYST. 

``` py
scan.set_setting(ra=0, dec=0, alt=60, date='2001-12-09')
```

Alternatively, you can set the setting by the initial datetime (UTC). 

``` py
scan.set_setting(ra=0, dec=0, datetime='2001-12-09 03:00:00')
```

You can save this data. There is an associated parameters file containing the parameters used to generate the scan pattern. By default, the data file is stored along with the parameters file in a folder in your CWD named after that scan pattern. 

```py
scan.to_csv('path_to_data_file.csv')
```

Afterwards, you can initialize a scan pattern using the path to that data file. 
```py
new_scan = CurvyPong('path_to_data_file.csv')
```
