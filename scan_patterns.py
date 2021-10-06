import math
from math import pi, sin, cos, tan, sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import pandas as pd
from scipy.optimize import fmin
from mpl_toolkits.axes_grid1 import make_axes_locatable

from datetime import timezone
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

ccatp_loc = EarthLocation(lat='-22d59m08.30s', lon='-67d44m25.00s', height=5611.8*u.m)

""" TODO
-- units and range
-- checking input parameters
"""

class ScanPattern:

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

        hour_angles = [hr - 24 if hr > 12 else hr for hr in hour_angles.to(u.hourangle).value]

        # populate dataframe
        self.df['az_coord'] = obs.az.deg
        self.df['alt_coord'] = obs.alt.deg
        self.df['az_vel'] = az_vel
        self.df['alt_vel'] = alt_vel
        self.df['az_acc'] = az_acc
        self.df['alt_acc'] = alt_acc
        self.df['hour_angle'] = hour_angles
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

    def plot(self, graphs=['coord']):
        
        if 'coord' in graphs:
            fig_coord, ax_coord = plt.subplots(1, 2)
            ax_coord[0].plot(self.df['x_coord']/3600, self.df['y_coord']/3600)
            ax_coord[0].set_aspect('equal', 'box')
            ax_coord[0].set(xlabel='Right Ascension (degrees)', ylabel='Declination (degrees)', title='RA/DEC')
            ax_coord[1].plot(self.df['az_coord'], self.df['alt_coord'])
            ax_coord[1].set_aspect('equal', 'box')
            ax_coord[1].set(xlabel='Azimuth (degrees)', ylabel='Altitude (degrees)', title='AZ/ALT (2001-12-09 ~60°ALT @ FYST)')
            fig_coord.tight_layout()
        
        if 'vel' in graphs:
            fig_vel, ax_vel = plt.subplots(2, 1, sharex=True, sharey=True)

            total_vel = np.sqrt(self.df['x_vel']**2 + self.df['y_vel']**2)
            ax_vel[0].plot(self.df['time_offset'], total_vel/3600, label='Total', c='black', ls='dashed', alpha=0.25)
            ax_vel[0].plot(self.df['time_offset'], self.df['x_vel']/3600, label='RA')
            ax_vel[0].plot(self.df['time_offset'], self.df['y_vel']/3600, label='DEC')
            ax_vel[0].legend(loc='upper right')
            ax_vel[0].set(xlabel='time offset (s)', ylabel='velocity (deg/s)', title='RA/DEC Velocity')
            ax_vel[0].grid()

            total_vel = np.sqrt(self.df['az_vel']**2 + self.df['alt_vel']**2)
            ax_vel[1].plot(self.df['time_offset'], total_vel, label='Total', c='black', ls='dashed', alpha=0.25)
            ax_vel[1].plot(self.df['time_offset'], self.df['az_vel'], label='AZ')
            ax_vel[1].plot(self.df['time_offset'], self.df['alt_vel'], label='ALT')
            ax_vel[1].legend(loc='upper right')
            ax_vel[1].set(xlabel='time offset (s)', ylabel='velocity (deg/s)', title='ALT/AZ Velocity (2001-12-09 ~60°ALT @ FYST)')
            ax_vel[1].grid()

            fig_vel.tight_layout()
        
        if 'acc' in graphs:
            fig_acc, ax_acc = plt.subplots(2, 1, sharex=True, sharey=True)

            total_acc = np.sqrt(self.df['x_acc']**2 + self.df['y_acc']**2)
            ax_acc[0].plot(self.df['time_offset'], total_acc/3600, label='Total', c='black', ls='dashed', alpha=0.25)
            ax_acc[0].plot(self.df['time_offset'], self.df['x_acc']/3600, label='RA')
            ax_acc[0].plot(self.df['time_offset'], self.df['y_acc']/3600, label='DEC')
            ax_acc[0].legend(loc='upper right')
            ax_acc[0].set(xlabel='time offset (s)', ylabel='acceleration (deg/s^2)', title='RA/DEC Acceleration')
            ax_acc[0].grid()

            total_acc = np.sqrt(self.df['az_acc']**2 + self.df['alt_acc']**2)
            ax_acc[1].plot(self.df['time_offset'], total_acc, label='Total', c='black', ls='dashed', alpha=0.25)
            ax_acc[1].plot(self.df['time_offset'], self.df['az_acc'], label='AZ')
            ax_acc[1].plot(self.df['time_offset'], self.df['alt_acc'], label='ALT')
            ax_acc[1].legend(loc='upper right')
            ax_acc[1].set(xlabel='time offset (s)', ylabel='acceleration (deg/s^2)', title='AZ/ALT Acceleration (2001-12-09 ~60°ALT @ FYST)')
            ax_acc[1].grid()

            fig_acc.tight_layout()

        if 'quiver' in graphs:
            fig_quiver, ax_quiver = plt.subplots(1, 2, sharex=True, sharey=True)
            subsample = 100
            endpoint = None

            # --- ACCELERATION ---

            # plot acc
            total_acc = np.sqrt(self.df['x_acc']**2 + self.df['y_acc']**2).to_numpy()/3600
            ax_quiver[0].plot(self.df['x_coord']/3600, self.df['y_coord']/3600, alpha=0.25, color='black')
            pcm = ax_quiver[0].quiver(
                self.df['x_coord'].to_numpy()[:endpoint:subsample]/3600, self.df['y_coord'].to_numpy()[:endpoint:subsample]/3600, 
                self.df['x_acc'].to_numpy()[:endpoint:subsample]/3600, self.df['y_acc'].to_numpy()[:endpoint:subsample]/3600,
                total_acc[:endpoint:subsample], #clim=(0, 1)
            )
            ax_quiver[0].set_aspect('equal', 'box')
            ax_quiver[0].set(xlabel='Map height [deg]', ylabel='Map width [deg]', title='RA/DEC acc [deg/s^2]')

            # colorbar acc
            divider = make_axes_locatable(ax_quiver[0])
            cax = divider.append_axes("right", size="3%", pad=0.5)
            fig_quiver.colorbar(pcm, cax=cax)

            # --- JERK ---

            # get jerk
            x_jerk = self._central_diff(self.df['x_acc'].to_numpy(), self.sample_interval)
            y_jerk = self._central_diff(self.df['y_acc'].to_numpy(), self.sample_interval)
            total_jerk = np.sqrt(x_jerk**2 + y_jerk**2)/3600

            # plot jerk
            ax_quiver[1].plot(self.df['x_coord']/3600, self.df['y_coord']/3600, alpha=0.25, color='black')
            pcm = ax_quiver[1].quiver(
                self.df['x_coord'].to_numpy()[:endpoint:subsample]/3600, self.df['y_coord'].to_numpy()[:endpoint:subsample]/3600, 
                x_jerk[:endpoint:subsample]/3600, y_jerk[:endpoint:subsample]/3600,
                total_jerk[:endpoint:subsample], #clim=(0, 1)
            )
            ax_quiver[1].set_aspect('equal', 'box')
            ax_quiver[1].set(xlabel='Map height [deg]', ylabel='Map width [deg]', title='RA/DEC jerk [deg/s^3]')

            # colorbar
            divider = make_axes_locatable(ax_quiver[1])
            cax = divider.append_axes("right", size="3%", pad=0.5)
            fig_quiver.colorbar(pcm, cax=cax)
            
            fig_quiver.tight_layout()


        plt.show()

    def hitmap(self, pixelpos_files, rot=0, max_acc=None, plate_scale=52, percent=1, grid_size=10):
        
        # remove points with high acceleration 
        total_azalt_acc = np.sqrt(self.df['az_acc']**2 + self.df['alt_acc']**2)

        if max_acc is None:
            x_coord = self.df['x_coord'].to_numpy() # arcsec
            y_coord = self.df['y_coord'].to_numpy() # arcsec
            rot_angle = np.radians(self.df['rot_angle'].to_numpy())
        else:
            mask = total_azalt_acc < 0.4
            x_coord = self.df.loc[mask, 'x_coord'].to_numpy()
            y_coord = self.df.loc[mask, 'y_coord'].to_numpy()
            rot_angle = np.radians(np.radians(self.df.loc[mask, 'rot_angle'].to_numpy()))

            x_coord_removed = self.df.loc[~mask, 'x_coord'].to_numpy()
            y_coord_removed = self.df.loc[~mask, 'y_coord'].to_numpy()
            rot_angle_removed = np.radians(self.df.loc[~mask, 'rot_angle'].to_numpy())
        
        # get pixel positions (convert meters->arcsec)
        x_pixel = np.array([])
        y_pixel = np.array([])

        for f in pixelpos_files:
            x, y = np.loadtxt(f, unpack=True)
            x_pixel = np.append(x_pixel, x)
            y_pixel = np.append(y_pixel, y)

        pixel_size = math.sqrt((x_pixel[0] - x_pixel[1])**2 + (y_pixel[0] - y_pixel[1])**2)
        print('pixel_size =', pixel_size)
        print('total number of detector pixels =', len(x_pixel))
        x_pixel = x_pixel/pixel_size*plate_scale # arcsec
        y_pixel = y_pixel/pixel_size*plate_scale # arcsec
        
        # define bin edges
        dist_from_center = np.sqrt(x_pixel**2 + y_pixel**2)
        det_max = max(dist_from_center)

        #x_max = math.ceil((max(abs(x_coord)) + det_max)/grid_size)*grid_size
        #y_max = math.ceil((max(abs(y_coord)) + det_max)/grid_size)*grid_size
        x_max = 5500
        y_max = 5500
        x_edges = np.arange(-x_max, x_max+1, grid_size)
        y_edges = np.arange(-y_max, y_max+1, grid_size)
        print('x max min =', x_edges[0], x_edges[-1])
        print('y max min =', y_edges[0], y_edges[-1])

        hist = np.zeros( (len(x_edges)-1 , len(y_edges)-1) )
        hist_removed = np.zeros( (len(x_edges)-1 , len(y_edges)-1) )

        # get only first x percent of scan
        last_sample = math.ceil(len(x_coord)*percent)

        # sort all positions with individual detector offset into a 2D histogram
        # FIXME rotation, optimization
        for x_off, y_off in zip(x_pixel, y_pixel):
            x_off_rot = x_off*np.cos(rot_angle[:last_sample] + rot) + y_off*np.sin(rot_angle[:last_sample] + rot)
            y_off_rot = -x_off*np.sin(rot_angle[:last_sample] + rot) + y_off*np.cos(rot_angle[:last_sample] + rot)
            hist_temp, _, _ = np.histogram2d(x_coord[:last_sample] + x_off_rot, y_coord[:last_sample] + y_off_rot, bins=[x_edges, y_edges])
            hist += hist_temp

            if not max_acc is None:
                x_off_rot_removed = x_off*np.cos(rot_angle_removed + rot) + y_off*np.sin(rot_angle_removed + rot)
                y_off_rot_removed = -x_off*np.sin(rot_angle_removed + rot) + y_off*np.cos(rot_angle_removed + rot)     
                hist_temp_rem, _, _ = np.histogram2d(x_coord_removed + x_off_rot_removed, y_coord_removed + y_off_rot_removed, bins=[x_edges, y_edges])
                hist_removed += hist_temp_rem

        print('shape:', np.shape(hist), '<->', len(x_edges), len(y_edges))

        # -- PLOTTING --

        fig = plt.figure(1)

        # plot histogram
        ax1 = plt.subplot2grid((3, 3), (0, 1), rowspan=2)
        pcm = ax1.imshow(hist.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], vmin=0, interpolation='nearest', origin='lower')
        ax1.set_aspect('equal', 'box')
        field = patches.Rectangle((-7200/2, -7000/2), width=7200, height=7000, linewidth=1, edgecolor='r', facecolor='none') #FIXME
        ax1.add_patch(field)
        subtitle = f'alt=30, max acc={max_acc}, pixel size={grid_size}'
        ax1.set(xlabel='x offset (arcsec)', ylabel='y offset (arcsec)')
        ax1.set_title('Kept hits per pixel\n'+subtitle, fontsize=12)
        ax1.scatter(x_coord[:last_sample], y_coord[:last_sample], color='r', s=0.001)
        ax1.axvline(x=0, c='black')
        ax1.axhline(y=0, c='black')
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("bottom", size="3%", pad=0.5)
        fig.colorbar(pcm, cax=cax1, orientation='horizontal')

        kept_hits = sum(hist.flatten())
        removed_hits = sum(hist_removed.flatten())
        print(f'{kept_hits}/{kept_hits + removed_hits}')
        textstr = f'{round(kept_hits/(kept_hits+removed_hits)*100, 2)}% hits kept'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.1, 0.9, textstr, transform=ax1.transAxes, bbox=props)

        # plot detector pixel positions
        subtitle4 = f'rot={rot}, pixel dist={round(pixel_size, 5)}, plate scale={plate_scale}'
        ax4 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, sharex=ax1, sharey=ax1)
        ax4.scatter(x_pixel*cos(rot) + y_pixel*sin(rot), -x_pixel*sin(rot) + y_pixel*cos(rot), s=0.01)
        ax4.set_aspect('equal', 'box')
        ax4.set(xlabel='x offset (arcsec)', ylabel='y offset (arcsec)')
        ax4.set_title('Detector Pixel Positions\n'+subtitle4, fontsize=12)
        divider4 = make_axes_locatable(ax4)
        cax4 = divider4.append_axes("bottom", size="3%", pad=0.5)
        cax4.axis('off')

        # bin line plot
        ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        bin_index = int(x_max/grid_size)
        y_values = hist[bin_index]
        ax3.plot(y_edges[:-1], y_values, label='Kept hits', drawstyle='steps')
        ax3.axvline(x=-7200/2, c='r') #FIXME
        ax3.axvline(x=7200/2, c='r') #FIXME
        ax3.set(ylabel='Hits/Pixel', xlabel='y offset (arcsec)')
        ax3.set_title('Hit count in x=0 to x=10 bin', fontsize=12)

        # removed points
        if not max_acc is None:
            ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2, sharex=ax1, sharey=ax1)
            pcm_removed = ax2.imshow(hist_removed.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], vmin=0, interpolation='nearest', origin='lower')
            ax2.set_aspect('equal', 'box')
            field_removed = patches.Rectangle((-7200/2, -7000/2), width=7200, height=7000, linewidth=1, edgecolor='r', facecolor='none') #FIXME
            ax2.add_patch(field_removed)
            ax2.set(xlabel='x offset (arcsec)', ylabel='y offset (arcsec)')
            ax2.set_title('Removed hits per pixel\n'+subtitle, fontsize=12)
            ax2.scatter(x_coord_removed, y_coord_removed, color='r', s=0.001)
            ax2.axvline(x=0, c='black')
            ax2.axhline(y=0, c='black')
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes("bottom", size="3%", pad=0.5)
            fig.colorbar(pcm_removed, cax=cax2, orientation='horizontal')

            y_values_removed = hist_removed[bin_index]
            ax3.plot(y_edges[:-1], y_values_removed, label='Removed hits', drawstyle='steps')
            ax3.plot(y_edges[:-1], y_values + y_values_removed, label='Total hits', drawstyle='steps', color='black')

        ax3.legend(loc='upper right')
        fig.tight_layout()
        plt.show()

    def to_csv(self, path, columns=None):
        if columns is None:
            self.df.to_csv(path)
        else:
            self.df.to_csv(path, columns=columns)


class CurvyPong(ScanPattern):
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


class Daisy(ScanPattern):
    def __init__(self, from_csv=None, speed=200, R0=120, y=0, Rt=120, Ra=100, acc=300, T=100, dt=0.005):
        
        if not from_csv is None:
            self.df = pd.read_csv(from_csv)
            self.sample_interval = self.df['time_offset'].iloc[0] if dt is None else dt
            return
        
        self.sample_interval = dt

        xval = np.array([]) # x,y arrays for plotting 
        yval = np.array([]) 
        x_vel = np.array([]) # speed array for plotting 
        y_vel = np.array([])

        (vx, vy) = (1.0, 0.0) # Tangent vector & start value
        (x, y) = (0.0, y) # Position vector & start value
        N = int(T/dt) + 1 # number of steps 
        R1 = min(R0, Ra) # Effective avoidance radius so Ra is not used if Ra > R0 

        print("# R0: ", R0 )
        print("# Rt: ", Rt)
        print("# R1: ", R1, "Ra: ", Ra) 
        print("# xstart: ", x, " ystart: ", y)
        print("# Time: ", T, " dt : ", dt) 
        print("# speed: ", speed, " acc: ", acc) 
        s0 = speed 
        speed = 0 # set start speed 

        for step in range(1, N): 
            speed += acc*dt # ramp up speed with acceleration acc 
            if speed >= s0: # to limit startup transients . Telescope 
                speed = s0  # has zero speed at startup. 

            r = sqrt(x*x + y*y) # compute distance from center 

            if r < R0: # Straight motion inside R0 
                x += vx*speed*dt 
                y += vy*speed*dt 
            else: 
                (xn,yn) = (x/r,y/r) # Compute unit radial vector 
                if (-xn*vx - yn*vy) > sqrt(1 - R1*R1/r/r): # If aming close to center 
                    x += vx*speed*dt # resume straight motion 
                    y += vy*speed*dt 
                else: 
                    if (-xn*vy + yn*vx) > 0: # Decide turning direction 
                        Nx = vy 
                        Ny = -vx 
                    else: 
                        Nx = -vy 
                        Ny = vx 

                    # Compute curved trajectory using serial exansion in step length s 
                    s = speed*dt 
                    x += (s - s*s*s/Rt/Rt/6)*vx + s*s/Rt/2*Nx 
                    y += (s - s*s*s/Rt/Rt/6)*vy + s*s/Rt/2*Ny 
                    vx += -s*s/Rt/Rt/2*vx + (s/Rt + s*s*s/Rt/Rt/Rt/6)*Nx 
                    vy += -s*s/Rt/Rt/2*vy + (s/Rt + s*s*s/Rt/Rt/Rt/6)*Ny 

            # Store result for plotting and statistics 
            xval = np.append(xval, x) 
            yval = np.append(yval, y) 
            x_vel = np.append(x_vel, speed*vx)
            y_vel = np.append(y_vel, speed*vy)

        # Compute arrays for plotting 
        ax = -2*xval[1: -1] + xval[0:-2] + xval[2:] # numerical acc in x 
        ay = -2*yval[1: -1] + yval[0:-2] + yval[2:] # numerical acc in y 
        x_acc = np.append(np.array([0]), ax/dt/dt)
        y_acc = np.append(np.array([0]), ay/dt/dt)
        x_acc = np.append(x_acc, 0)
        y_acc = np.append(y_acc, 0)
    
        #rval = np.array(np.sqrt(xval**2 + yval**2))

        self.df = pd.DataFrame({
            'time_offset': np.arange(0, T, dt), 
            'x_coord': xval, 'y_coord': yval, 
            'x_vel': x_vel, 'y_vel': y_vel,
            'x_acc': x_acc, 'y_acc': y_acc 
        })


def elevation_offset(dist=1.7):
    dist = math.radians(dist)

    # create bins
    bins = 1000
    bins_a = np.linspace(0, math.radians(80), bins)
    bins_theta1 = np.linspace(0, 2*pi, bins)
    a_grid, theta1_grid = np.meshgrid(bins_a, bins_theta1)

    # calculate
    da_grid = np.sin(a_grid)*cos(dist) + sin(dist)*np.cos(a_grid)*np.sin(theta1_grid + a_grid)
    da_grid = np.arcsin(da_grid) - a_grid

    # unit conversion
    da_grid = np.degrees(da_grid)
    a_grid = np.degrees(a_grid)
    theta1_grid = np.degrees(theta1_grid)

    # plot
    fig, ax = plt.subplots(1, 1)
    pcm = ax.contourf(a_grid, theta1_grid, da_grid, levels=np.arange(-2, 2.1, 0.2), cmap='RdBu')
    fig.colorbar(pcm, ax=ax)
    ax.set(xlabel='Boresight elevation (deg)', ylabel='$θ_1$ (deg)', title=f'Optics tube offset: Elevation')
    plt.show()

def azimuth_offset(dist=1.7):
    dist = math.radians(dist)

    # create bins
    bins = 1000
    bins_a = np.linspace(0, math.radians(80), bins)
    bins_theta1 = np.linspace(0, 2*pi, bins)
    a_grid, theta1_grid = np.meshgrid(bins_a, bins_theta1)
    A_grid = np.zeros((bins, bins))

    # calculate
    sinA1_grid = np.cos(a_grid)*np.sin(A_grid)*cos(dist) + np.cos(A_grid)*np.cos(theta1_grid + a_grid)*sin(dist) - np.sin(a_grid)*np.sin(A_grid)*sin(dist)*np.sin(theta1_grid + a_grid)
    cosA1_grid = np.cos(a_grid)*np.cos(A_grid)*cos(dist) - np.sin(A_grid)*np.cos(theta1_grid + a_grid)*sin(dist) - np.sin(a_grid)*np.cos(A_grid)*sin(dist)*np.sin(theta1_grid + a_grid)
    A1_grid = np.arctan2(sinA1_grid, cosA1_grid)

    # unit conversions
    a_grid = np.degrees(a_grid)
    theta1_grid = np.degrees(theta1_grid)
    A1_grid = np.degrees(A1_grid)

    # plot
    fig, ax = plt.subplots(1, 1)
    pcm = ax.contourf(a_grid, theta1_grid, A1_grid, levels=np.arange(-10, 10.1, 1), cmap='RdBu')
    fig.colorbar(pcm, ax=ax)
    ax.set(xlabel='Boresight elevation (deg)', ylabel='$θ_1$ (deg)', title=f'Optics tube offset: Azimuth')
    plt.show()

if __name__ == '__main__':
    #scan = CurvyPong(width=7200, height=7000, spacing=500, sample_interval=0.002, velocity=1000, angle=0, num_terms=5)
    #scan.set_setting(ra=0, dec=0, alt=60, date='2001-12-09')
    #scan.to_csv('curvy_pong_ra0dec0alt60_20011209.csv')
    scan = CurvyPong(from_csv='curvy_pong_ra0dec0alt30_20011209.csv', sample_interval=0.002)
    
    #scan.hitmap(pixelpos_files=['pixelpos1.txt', 'pixelpos2.txt', 'pixelpos3.txt'], rot=0, max_acc=0.4, plate_scale=26, percent=1, grid_size=10)
    scan.plot(['quiver'])