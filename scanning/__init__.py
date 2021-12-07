from astropy.coordinates import EarthLocation
import astropy.units as u
import numpy as np

# location of FYST
FYST_LOC = EarthLocation(lat='-22d59m08.30s', lon='-67d44m25.00s', height=5611.8*u.m)

def _central_diff(a, h=None, time=None):

    # get dt
    if h is None:
        h = time[1] - time[0]

    # get derivative 
    a = np.array(a)
    len_a = len(a)

    new_a = np.empty(len_a)
    new_a[0] = (a[1] - a[0])/h
    new_a[1:-1] = (a[2:len_a] - a[0:len_a-2])/(2*h)
    new_a[-1] = (a[-1] - a[-2])/h

    return new_a