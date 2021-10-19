from scanning.scan_patterns import Daisy, CurvyPong
from line_profiler import LineProfiler
import astropy.units as u
import io


#scan = CurvyPong(num_terms=5, width=2, height=7000*u.arcsec, spacing='500 arcsec', velocity='1000 arcsec/s', sample_interval=0.2)
scan = Daisy(speed=1/3*u.deg/u.s, acc=1*u.deg/u.s/u.s, R0=1400*u.arcsec, Rt=1400*u.arcsec, Ra=1200*u.arcsec, T=360*u.s, sample_interval=1/40*u.s)
scan.set_setting(ra=0, dec=0, alt=30, date='2001-12-09')
#scan = CurvyPong('ra0dec0alt30_20011209.csv')

"""s = io.StringIO()
lp = LineProfiler()

lp.add_function(scan._generate_hitmap)
lp_wrapper = lp(scan.hitmap)
lp_wrapper(plate_scale=52*u.arcsec, pixel_scale=10*u.arcsec)

lp.print_stats(stream=s, output_unit=1)
with open('delete/_.txt', 'w+') as f:
    f.write(s.getvalue())"""

scan.plot(['acc'])
scan.hitmap(plate_scale=52*u.arcsec, pixel_scale=10*u.arcsec, max_acc=0.2*u.deg/u.s/u.s)