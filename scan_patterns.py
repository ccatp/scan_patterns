import math
from math import pi, sin, cos, tan
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from scipy.optimize import fmin

from datetime import timezone, tzinfo
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

ccatp_loc = EarthLocation(lat='-22d59m08.30s', lon='-67d44m25.00s', height=5611.8*u.m)

""" TODO
-- units
-- checking input parameters
-- flexible plotting
-- hour angle and parallactic angle is correct (remove if...)
"""

class CurvyPong:

    def __init__(self, from_csv=None, width=None, height=None, spacing=None, sample_interval=0.002, velocity=1000, angle=0, num_terms=5):

        if not from_csv is None:
            self.df = pd.read_csv(from_csv)
            self.sample_interval = self.df['time_offset'].iloc[0] if sample_interval is None else sample_interval
            return

        # Record parameters
        self.sample_interval = sample_interval
        
        # Determine number of vertices (reflection points) along each side of the
        # box which satisfies the common-factors criterion and the requested size / spacing    

        vert_spacing = math.sqrt(2)*spacing
        x_numvert = math.ceil(width/vert_spacing)
        y_numvert = math.ceil(height/vert_spacing)
 
        if x_numvert%2 == y_numvert%2:
            if x_numvert >= y_numvert:
                y_numvert += 1
            else:
                x_numvert += 1

        num_vert = [x_numvert, y_numvert]
        most_i = num_vert.index(max(x_numvert, y_numvert))
        least_i = num_vert.index(min(x_numvert, y_numvert))

        while math.gcd(num_vert[most_i], num_vert[least_i]) != 1:
            num_vert[most_i] += 2
        
        x_numvert = num_vert[0]
        y_numvert = num_vert[1]
        assert(math.gcd(x_numvert, y_numvert) == 1)
        assert((x_numvert%2 == 0 and y_numvert%2 == 1) or (x_numvert%2 == 1 and y_numvert%2 == 0))

        # Calculate the approximate periods by assuming a Pong scan with
        # no rounding at the corners. Average the x- and y-velocities
        # in order to determine the period in each direction, and the
        # total time required for the scan.

        vavg = velocity
        peri_x = x_numvert * vert_spacing * 2 / vavg
        peri_y = y_numvert * vert_spacing * 2 / vavg
        period = x_numvert * y_numvert * vert_spacing * 2 / vavg

        pongcount = math.ceil(period/sample_interval)
        amp_x = x_numvert * vert_spacing / 2
        amp_y = y_numvert * vert_spacing / 2
        
        # Calculate the grid positions and apply rotation angle. Load
        # data into a dataframe.

        t_count = 0
        time_offset = []
        x_coord = []
        y_coord = []
        x_vel = []
        y_vel = []
        x_acc = []
        y_acc = []

        for i in range(pongcount + 1):
            tx, ttx, tttx = self._fourier_expansion(num_terms, amp_x, t_count, peri_x)
            ty, tty, ttty = self._fourier_expansion(num_terms, amp_y, t_count, peri_y)

            x_coord.append(tx*cos(angle) - ty*sin(angle))
            y_coord.append(tx*sin(angle) + ty*cos(angle))
            x_vel.append(ttx*cos(angle) - tty*sin(angle))
            y_vel.append(ttx*sin(angle) + tty*cos(angle))
            x_acc.append(tttx*cos(angle) - ttty*sin(angle))
            y_acc.append(tttx*sin(angle) + ttty*cos(angle))

            time_offset.append(t_count)
            t_count += sample_interval
        
        self.df = pd.DataFrame({
            'time_offset': time_offset, 
            'x_coord': x_coord, 'y_coord': y_coord, 
            'x_vel': x_vel, 'y_vel': y_vel,
            'x_acc': x_acc, 'y_acc': y_acc
        })
    
    def set_setting(self, ra, dec, alt, date, location=ccatp_loc):
        """ 
        Given an object (ra, dec) and a desired alt, find the az/alt coordinates
        for a specific date and location. Formulas from http://www.stargazing.net/kepler/altaz.html
        
        Later, possibly, include option of given object (ra, dec) and a desired datetime. 

        ra (hours), dec (deg), alt (deg), date (str e.g. 'YYYY-MM-DD')
        """

        # GIVEN: RA, DEC, ALT, DATE

        # unit conversions
        ra = ra*u.hourangle
        dec = dec*u.deg
        alt = alt*u.deg

        # determine possible range of altitudes for the givens

        # determine possible hour angles
        lat = ccatp_loc.lat
        cos_ha = (sin(alt.to(u.rad).value) - sin(dec.to(u.rad).value)*sin(lat.to(u.rad).value)) / (cos(dec.to(u.rad).value)*cos(lat.to(u.rad).value))
        ha = math.acos(cos_ha)*u.rad

        # choose hour angle going up (FIXME to make more flexible)
        ha1_alt = self._alt(dec, lat, ha)
        ha1_delta = self._alt(dec, lat, ha + 1*u.deg) - ha1_alt
        ha1_up = True if ha1_delta > 0 else False
        print(f'hour angle = {ha.to(u.hourangle).value}, altitude = {ha1_alt.to(u.deg).value}, delta = {ha1_delta}')

        ha2_alt = self._alt(dec, lat, -ha)
        ha2_delta = self._alt(dec, lat, -ha + 1*u.deg) - ha2_alt
        ha2_up = True if ha2_delta > 0 else False
        print(f'hour angle = {-ha.to(u.hourangle).value}, altitude = {ha2_alt.to(u.deg).value}, delta = {ha2_delta}')

        if (ha1_up and ha2_up) or (not ha1_up and not ha2_up):
            raise Exception('An issue arised regarding hour angle and altitude.')
        elif ha2_up:
            ha = -ha

        # find ut (universal time after midnight of chosen date) 
        lon = ccatp_loc.lon
        time0 = Time(date, scale='utc')
        num_days = (time0 - Time(2000, format='jyear')).value # days from J2000
        lst = ha + ra
        ut = (lst.to(u.deg).value - 100.46 - 0.98564*num_days - lon.to(u.deg).value)/15
        time0 = pd.Timestamp(date, tzinfo=timezone.utc) + pd.Timedelta(ut%24, 'hour')

        # apply datetime to time_offsets
        print(f'start time = {time0.strftime("%Y-%m-%d %H:%M:%S.%f%z")}')
        df_datetime = pd.to_timedelta(self.df['time_offset'], unit='sec') + time0
        self.df.insert(0, 'datetime', df_datetime)

        # GENERAL 

        # convert to altitude/azimuth
        x_coord = self.df['x_coord']/3600 + ra.to(u.deg).value
        y_coord = self.df['y_coord']/3600 + dec.to(u.deg).value
        obs = SkyCoord(ra=x_coord*u.deg, dec=y_coord*u.deg, frame='icrs')
        print('converting to altitude/azimuth, this may take some time...')
        obs = obs.transform_to(AltAz(obstime=df_datetime, location=location))
        print('...converted!')

        # get velocity and acceleration in alt/az
        az_vel = self._central_diff(obs.az.deg, self.sample_interval)
        alt_vel = self._central_diff(obs.alt.deg, self.sample_interval)
        az_acc = self._central_diff(az_vel, self.sample_interval)
        alt_acc = self._central_diff(alt_vel, self.sample_interval)

        # get parallactic angle and rotation angle
        obs_time = Time(df_datetime, scale='utc', location=location)
        lst = obs_time.sidereal_time('apparent')
        hour_angles = lst - ra
        para = np.degrees(
            np.arctan2( 
                np.sin(hour_angles.to(u.rad).value), 
                cos(dec.to(u.rad).value)*tan(ccatp_loc.lat.rad) - sin(dec.to(u.rad).value)*np.cos(hour_angles.to(u.rad).value) 
            )
        )
        rot = para + obs.alt.deg

        # populate dataframe
        self.df['az_coord'] = obs.az.deg
        self.df['alt_coord'] = obs.alt.deg
        self.df['az_vel'] = az_vel
        self.df['alt_vel'] = alt_vel
        self.df['az_acc'] = az_acc
        self.df['alt_acc'] = alt_acc
        self.df['hour_angle'] = hour_angles.to(u.hourangle).value
        self.df['para_angle'] = para
        self.df['rot_angle'] = rot

    def _alt(self, dec, lat, ha):
        sin_alt = sin(dec.to(u.rad).value)*sin(lat.to(u.rad).value) + cos(dec.to(u.rad).value)*cos(lat.to(u.rad).value)*cos(ha.to(u.rad).value)
        return math.asin(sin_alt)*u.rad

    def _fourier_expansion(self, num_terms, amp, t_count, peri):
        N = num_terms*2 - 1
        a = (8*amp)/(pi**2)
        b = 2*pi/peri

        position = 0
        velocity = 0
        acc = 0
        for n in range(1, N+1, 2):
            c = math.pow(-1, (n-1)/2)/n**2 
            position += c * sin(b*n*t_count)
            velocity += c*n * cos(b*n*t_count)
            acc      += c*n**2 * sin(b*n*t_count)

        position *= a
        velocity *= a*b
        acc      *= -a*b**2
        return position, velocity, acc

    def _central_diff(self, a, h):
        new_a = [(a[1] - a[0])/h]
        for i in range(1, len(a)-1):
            new_a.append( (a[i+1] - a[i-1])/(2*h) )
        new_a.append( (a[-1] - a[-2])/h )
        return np.array(new_a)

    def plot(self, graphs):
        # [('x_coord', 'y_coord'), ('az_coord', 'alt_coord')]
        # [[('time_offset', 'x_vel'), ('time_offset', 'y_vel')], [('time_offset', 'az_vel'), ('time_offset', 'alt_vel')]]
        # [[('time_offset', 'x_acc'), ('time_offset', 'y_acc')], [('time_offset', 'az_acc'), ('time_offset', 'alt_acc')]]
        # [('hour_angle', 'para_angle'), ('hour_angle', 'rot_angle')]

        graph_shape = np.shape(graphs)

        # more than one row
        if len(graph_shape) == 3:
            nrows = graph_shape[0]
            ncols = graph_shape[1]
            fig, axes = plt.subplots(nrows, ncols)

            for row, graph_row in zip(axes, graphs):
                for ax, graph in zip(row, graph_row):
                    ax.plot(self.df[graph[0]], self.df[graph[1]])
                    ax.set(xlabel=graph[0], ylabel=graph[1])

        # one row
        elif len(graph_shape) == 2:
            ncols = graph_shape[0]
            fig, axes = plt.subplots(1, ncols)

            for ax, graph in zip(axes, graphs):
                ax.plot(self.df[graph[0]], self.df[graph[1]])
                ax.set(xlabel=graph[0], ylabel=graph[1])

        fig.tight_layout()
        plt.show()

    def hitmap(self, pixelpos_files, rot=0, max_acc=None, plate_scale=52, percent=1, grid_size=10):
        
        # remove points with high acceleration 
        total_azalt_acc = np.sqrt(self.df['az_acc']**2 + self.df['alt_acc']**2)

        if max_acc is None:
            x_coord = self.df['x_coord'].to_numpy() # arcsec
            y_coord = self.df['y_coord'].to_numpy() # arcsec
        else:
            mask = total_azalt_acc < 0.4
            x_coord = self.df.loc[mask, 'x_coord'].to_numpy()
            y_coord = self.df.loc[mask, 'y_coord'].to_numpy()

            x_coord_removed = self.df.loc[~mask, 'x_coord'].to_numpy()
            y_coord_removed = self.df.loc[~mask, 'y_coord'].to_numpy()
        
        # get pixel positions (convert meters->arcsec)
        x_pixel = np.array([])
        y_pixel = np.array([])

        for f in pixelpos_files:
            x, y = np.loadtxt(f, unpack=True)
            x_pixel = np.append(x_pixel, x)
            y_pixel = np.append(y_pixel, y)

        pixel_size = math.sqrt((x_pixel[0] - x_pixel[1])**2 + (y_pixel[0] - y_pixel[1])**2)
        print('pixel_size =', pixel_size)
        x_pixel = x_pixel/pixel_size*plate_scale # arcsec
        y_pixel = y_pixel/pixel_size*plate_scale # arcsec
        
        # define bin edges
        dist_from_center = np.sqrt(x_pixel**2 + y_pixel**2)
        det_max = max(dist_from_center)
        print('det_max =', det_max)

        x_max = math.ceil((max(abs(x_coord)) + det_max)/grid_size)*grid_size
        y_max = math.ceil((max(abs(y_coord)) + det_max)/grid_size)*grid_size
        x_edges = np.arange(-x_max, x_max+1, grid_size)
        y_edges = np.arange(-y_max, y_max+1, grid_size)
        print(x_edges[0], x_edges[-1])
        print(y_edges[0], y_edges[-1])

        hist = np.zeros( (len(x_edges)-1 , len(y_edges)-1) )

        # get only first x percent of scan
        last_sample = math.ceil(len(x_coord)*percent)

        # sort all positions with individual detector offset into a 2D histogram
        for x_off, y_off in zip(x_pixel, y_pixel):
            x_off_rot = x_off*cos(rot) + y_off*sin(rot)
            y_off_rot = -x_off*sin(rot) + y_off*cos(rot)

            hist_temp, _, _ = np.histogram2d(x_coord[:last_sample] + x_off_rot, y_coord[:last_sample] + y_off_rot, bins=[x_edges, y_edges])
            hist += hist_temp

        print('shape:', np.shape(hist), '<->', len(x_edges), len(y_edges))
        print('total hits:', sum(hist.flatten()))

        # -- PLOTTING --

        fig = plt.figure(1)

        # plot histogram
        ax1 = plt.subplot(2, 2, 1)
        pcm = ax1.imshow(hist.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], vmin=0, interpolation='nearest', origin='lower')
        fig.colorbar(pcm, ax=ax1)
        ax1.set_aspect('equal', 'box')
        field = patches.Rectangle((-7200/2, -7000/2), width=7200, height=7000, linewidth=1, edgecolor='r', facecolor='none') #FIXME
        ax1.add_patch(field)
        subtitle = f'rot={rot}, max_acc={max_acc}, plate_scale={plate_scale} \npixel_size={grid_size} (grid_size={np.shape(hist)}px)'
        ax1.set(xlabel='x offset (arcsec)', ylabel='y offset (arcsec)', title='Hits per pixel\n' + subtitle)
        ax1.scatter(x_coord[:last_sample], y_coord[:last_sample], color='r', s=0.001)

        # removed points
        if not max_acc is None:
            hist_removed = np.zeros( (len(x_edges)-1 , len(y_edges)-1) )
            for x_off, y_off in zip(x_pixel, y_pixel):
                x_off_rot = x_off*cos(rot) + y_off*sin(rot)
                y_off_rot = -x_off*sin(rot) + y_off*cos(rot)

                hist_temp, _, _ = np.histogram2d(x_coord_removed + x_off_rot, y_coord_removed + y_off_rot, bins=[x_edges, y_edges])
                hist_removed += hist_temp

            ax2 = plt.subplot(2, 2, 2)
            pcm_removed = ax2.imshow(hist_removed.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], vmin=0, interpolation='nearest', origin='lower')
            fig.colorbar(pcm_removed, ax=ax2)
            ax2.set_aspect('equal', 'box')
            field_removed = patches.Rectangle((-7200/2, -7000/2), width=7200, height=7000, linewidth=1, edgecolor='r', facecolor='none') #FIXME
            ax2.add_patch(field_removed)
            ax2.set(xlabel='x offset (arcsec)', ylabel='y offset (arcsec)', title='Removed hits per pixel\n' + subtitle)
            ax2.scatter(x_coord_removed, y_coord_removed, color='r', s=0.001)

        # bin line plot
        ax3 = plt.subplot(2, 1, 2)
        bin_index = int(x_max/grid_size)
        y_values = hist[bin_index]
        ax3.plot(y_edges[:-1], y_values, label='Hits', drawstyle='steps')
        y_values_removed = hist_removed[bin_index]
        ax3.plot(y_edges[:-1], y_values_removed, label='Removed hits', drawstyle='steps')
        ax3.axvline(x=-7200/2, c='r') #FIXME
        ax3.axvline(x=7200/2, c='r') #FIXME
        ax3.set(ylabel='Hits/Pixel', xlabel='y offset (arcsec)', title='Hit count in x=0 to x=10 bin')
        ax3.legend(loc='upper right')

        # plot detector pixel positions
        #fig_det, ax_det = plt.subplots(1, 1)
        #ax_det.scatter(x_pixel*cos(rot) + y_pixel*sin(rot), -x_pixel*sin(rot) + y_pixel*cos(rot), s=0.5)
        #ax_det.set(xlabel='x offset (arcsec)', ylabel='y offset (arcsec)', title='Detector Pixel Positions')
        #ax_det.set_aspect('equal', 'box')
        #fig_det.tight_layout()

        fig.tight_layout()
        plt.show()

    def to_csv(self, path, columns=None):
        if columns is None:
            self.df.to_csv(path)
        else:
            self.df.to_csv(path, columns=columns)
        
if __name__ == '__main__':
    #scan = CurvyPong(width=7200, height=7000, spacing=500, sample_interval=0.002, velocity=1000, angle=0, num_terms=5)
    #scan.set_setting(ra=0, dec=0, alt=60, date='2001-12-09')
    #scan.to_csv('curvy_pong_ra0dec0alt60_20011209.csv')

    scan = CurvyPong(from_csv='curvy_pong_ra0dec0alt30_20011209.csv', sample_interval=0.002)
    scan.hitmap(pixelpos_files=['pixelpos1.txt', 'pixelpos2.txt', 'pixelpos3.txt'], rot=0, max_acc=0.4, plate_scale=26, percent=1, grid_size=10)
    #scan.plot([('x_coord', 'y_coord'), ('az_coord', 'alt_coord')])