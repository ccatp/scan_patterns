import os
import warnings 
import sys
import cProfile
import pstats
import io

import math
from math import pi, sin, cos, tan, sqrt, radians, degrees, ceil
from astropy.units.equivalencies import pixel_scale, plate_scale
import numpy as np
import pandas as pd
from datetime import timezone

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

FYST_LOC = EarthLocation(lat='-22d59m08.30s', lon='-67d44m25.00s', height=5611.8*u.m)

# FIXME everything in deg s --> units in params.csv, arcsec for coord

class ScanPattern():

    def __init__(self, data_csv, param_csv=None):

        # find default parameter csv file or supply the parameters
        if param_csv is None:
            if os.path.isfile(os.path.join(self.default_folder, self.default_param_csv)):
                param_csv = os.path.join(self.default_folder, self.default_param_csv)
            elif os.path.isfile(self.default_param_csv):
                param_csv = self.default_param_csv
            else:
                raise ValueError('No file_path to param_csv given and none found in current working directory.')

        # check data_csv in cwd as well as default folder
        if not os.path.isfile(data_csv) and os.path.isfile(os.path.join(self.default_folder, data_csv)):
            data_csv = os.path.join(self.default_folder, data_csv)
        
        self.data = pd.read_csv(data_csv, index_col=False)
        self.params = pd.read_csv(param_csv, index_col='file_name').loc[os.path.basename(data_csv)].to_dict()

    @u.quantity_input(dec='angle', lat='angle', ha='angle')
    def _find_altitude(self, dec, lat, ha):
        sin_alt = sin(dec.to(u.rad).value)*sin(lat.to(u.rad).value) + cos(dec.to(u.rad).value)*cos(lat.to(u.rad).value)*cos(ha.to(u.rad).value)
        return math.asin(sin_alt)*u.rad

    def _central_diff(self, a):
        h = self.params['sample_interval']
        new_a = [(a[1] - a[0])/h]
        for i in range(1, len(a)-1):
            new_a.append( (a[i+1] - a[i-1])/(2*h) )
        new_a.append( (a[-1] - a[-2])/h )
        return np.array(new_a)

    def to_csv(self, data_csv, param_csv=None, with_param_file=True):
        """
        Save generated data into a csv file and its associated parameters. 

        Parameters
        -----------------
        data_csv : str
            File path you want to save data into. 
        param_csv : str, defaults to finding/making a default paramter file 
            File path to parameter file 
        with_param_file : bool, defaults to True
            Whether to place the data file in the same directory as param_csv (True) or cwd (False)
        """

        common_path = os.path.join(self.default_folder, self.default_param_csv)

        if not data_csv.endswith('.csv'):
            warnings.warn('Warning: data_csv does not end with .csv')

        if param_csv is None:

            # check if param file already exists
            if os.path.isfile(self.default_param_csv):
                param_csv = self.default_param_csv
            elif os.path.isfile(common_path):
                param_csv = common_path
           
            # make a new param file
            else:
                if not os.path.isdir(self.default_folder):
                    os.mkdir(self.default_folder)

                pd.DataFrame(
                    columns=['file_name'] 
                ).to_csv(common_path, index=False)

                param_csv = common_path
                print(f'Created new parameter file: {param_csv}')
        
        if with_param_file:
            if not os.path.dirname(param_csv) == os.path.dirname(data_csv):
                data_csv = os.path.join(os.path.dirname(param_csv), data_csv)
            
        # update param file
        param_df = pd.read_csv(param_csv, index_col='file_name')
        base_name = os.path.basename(data_csv)
        for p in self.params.keys():
            param_df.loc[base_name, p] = self.params[p]
        param_df.to_csv(param_csv)
        print(f'Updated parameter file {param_csv}')

        # save data file
        self.data.to_csv(data_csv, index=False)
        print(f'Saved data to {data_csv}')    

    def set_setting(self, **kwargs):
        """ 
        Finding the AZ/ALT coordinates given a specific set of parameters. 
        Some formulas from http://www.stargazing.net/kepler/altaz.html
        
        Possible Inputs:
            ra, dec, alt, date, and location (although alt is limited)
            ra, dec, datetime, and location

        Parameters
        ---------------------------------
        **kwargs
        ra : angle-like
            Right ascension of object
            If no units specified, defaults to deg
        dec : angle-like
            Declination of object
            If no units specified, defaults to deg
        location : astropy.coordinates.EarthLocation, defaults to FYST #FIXME add option to provide lat/lon/height
            Location of observation/telescope
        
        alt : angle-like, optional
            Desired approximate initial altitude
            If no units specified, defaults to deg
        date : str or date-like, optional 
            Initial date of observation (UTC) #FIXME add option to specify timezone'
        moving_up : bool, defaults to True
            Whether to choose path such that object is moving up in altitude

        datetime : str or datetime-like, optional
            Initial datetime of observation (UTC)
        """

        ra = u.Quantity(kwargs.pop('ra'), u.deg).value
        dec = u.Quantity(kwargs.pop('dec'), u.deg).value
        location = kwargs.pop('location', FYST_LOC)

        self.params['ra'] = ra
        self.params['dec'] = dec
        self.params['lat'] = location.lat.to(u.deg).value
        self.params['lon'] = location.lon.to(u.deg).value
        self.params['location_height'] = location.height.to(u.m).value
 
        # ------ GIVEN: RA AND DATE --------

        if 'alt' in kwargs.keys():
            ra = radians(ra)
            dec = radians(dec)

            alt = u.Quantity(kwargs.pop('alt'), u.deg).to(u.rad).value
            moving_up = kwargs.pop('moving_up', True)
            date = kwargs.pop('date') #FIXME check that it's date, not datetime

            if bool(kwargs):
                raise ValueError(f'Additional arguments supplied: {kwargs}')

            # determine possible hour angles
            lat = location.lat.to(u.rad).value
            cos_ha = (sin(alt) - sin(dec)*sin(lat)) / (cos(dec)*cos(lat))
            try:
                ha = math.acos(cos_ha)
            except ValueError as e:
                raise ValueError('Altitude is not possible at provided RA, declination, and latitude.') from e
                # FIXME list range of possible altitudes

            # choose hour angle
            ha1_delta = self._find_altitude(dec*u.rad, lat*u.rad, ha*u.rad + 1*u.deg) - self._find_altitude(dec*u.rad, lat*u.rad, ha*u.rad)
            ha1_up = True if ha1_delta > 0 else False

            ha2_delta = self._find_altitude(dec*u.rad, lat*u.rad, -ha*u.rad + 1*u.deg) - self._find_altitude(dec*u.rad, lat*u.rad, -ha*u.rad)
            ha2_up = True if ha2_delta > 0 else False
            assert(ha1_up != ha2_up)

            if (moving_up and ha2_up) or (not moving_up and ha1_up):
                ha = -ha 

            # find ut (universal time after midnight of chosen date) 
            lon = location.lon.to(u.rad).value
            time0 = Time(date, scale='utc')
            num_days = (time0 - Time(2000, format='jyear')).value # days from J2000
            lst = ha + ra
            ut = (degrees(lst) - 100.46 - 0.98564*num_days - degrees(lon))/15
            time0 = pd.Timestamp(date, tzinfo=timezone.utc) + pd.Timedelta(ut%24, 'hour')

            ra = degrees(ra)
            dec = degrees(dec)
        
        # --------- GIVEN: DATETIME --------
        
        elif 'datetime' in kwargs.keys():
            time0 = Time(kwargs.pop('datetime'), scale='utc')

            if bool(kwargs):
                raise ValueError(f'Additional arguments supplied: {kwargs}')

        # -----------------------------------
        # ----------- GENERAL ---------------
        # -----------------------------------

        # apply datetime to time_offsets
        self.params['time0'] = time0.strftime("%Y-%m-%d %H:%M:%S.%f%z")
        print(f'start time = {time0.strftime("%Y-%m-%d %H:%M:%S.%f%z")}')
        df_datetime = pd.to_timedelta(self.data['time_offset'], unit='sec') + time0

        # convert to altitude/azimuth
        x_coord = self.data['x_coord'] + ra
        y_coord = self.data['y_coord'] + dec
        obs = SkyCoord(ra=x_coord*u.deg, dec=y_coord*u.deg, frame='icrs')
        print('Converting to altitude/azimuth, this may take some time...')
        obs = obs.transform_to(AltAz(obstime=df_datetime, location=location))
        print('...Converted!')

        # get velocity and acceleration and jerk in alt/az
        az_vel = self._central_diff(obs.az.deg)
        alt_vel = self._central_diff(obs.alt.deg)
        az_acc = self._central_diff(az_vel)
        alt_acc = self._central_diff(alt_vel)
        az_jerk = self._central_diff(az_acc)
        alt_jerk = self._central_diff(alt_acc)

        # get parallactic angle and rotation angle
        obs_time = Time(df_datetime, scale='utc', location=location)
        lst = obs_time.sidereal_time('apparent').deg
        hour_angles = lst - ra

        para = np.degrees(
            np.arctan2( 
                np.sin(np.radians(hour_angles)), 
                cos(radians(dec))*tan(location.lat.rad) - sin(radians(dec))*np.cos(np.radians(hour_angles)) 
            )
        )
        rot = para + obs.alt.deg

        hour_angles = [hr - 24 if hr > 12 else hr for hr in hour_angles*u.deg.to(u.hourangle)]

        # populate dataframe
        self.data['az_coord'] = obs.az.deg
        self.data['alt_coord'] = obs.alt.deg
        self.data['az_vel'] = az_vel
        self.data['alt_vel'] = alt_vel
        self.data['az_acc'] = az_acc
        self.data['alt_acc'] = alt_acc
        self.data['az_jerk'] = az_jerk
        self.data['alt_jerk'] = alt_jerk
        self.data['hour_angle'] = hour_angles # in hourangle 
        self.data['para_angle'] = para
        self.data['rot_angle'] = rot

    def _generate_hitmap(self, x_coord, y_coord, rot_angle, **kwargs):
        x_edges = kwargs['x_edges']
        y_edges = kwargs['y_edges']
        x_pixel = kwargs['x_pixel']
        y_pixel = kwargs['y_pixel']
        rot = kwargs['rot']
        
        num_ts = len(x_coord)
        print('number of timestamps:', num_ts)

        num_detectors = len(x_pixel)
        print('total number of detector pixels =', num_detectors)

        # sort all positions with individual detector offset into a 2D histogram

        MEM_LIMIT = 8*10**7 
        chunk_ts = math.floor(MEM_LIMIT/num_detectors)
        hist = np.zeros((len(x_edges)-1 , len(y_edges)-1))
        rot = radians(rot)

        for chunk in range(ceil(num_ts/chunk_ts)):

            if (chunk+1)*chunk_ts <= num_ts:
                all_x_coords = np.zeros(chunk_ts*num_detectors)
                all_y_coords = np.zeros(chunk_ts*num_detectors)
                last_sample = chunk_ts
            else:
                all_x_coords = np.zeros((num_ts - chunk*chunk_ts)*num_detectors)
                all_y_coords = np.zeros((num_ts - chunk*chunk_ts)*num_detectors)
                last_sample = num_ts - chunk*chunk_ts

            start = chunk*chunk_ts
            end = (chunk+1)*chunk_ts
            print('last_sample:', last_sample)
            print('start:', start, 'end:', end)
            for i, x_coord1, y_coord1, rot1 in zip(range(0, last_sample), x_coord[start:end], y_coord[start:end], np.radians(rot_angle[start:end])):
                all_x_coords[i*num_detectors: (i+1)*num_detectors] = x_coord1 + x_pixel*cos(rot1 + rot) + y_pixel*sin(rot1 + rot)
                all_y_coords[i*num_detectors: (i+1)*num_detectors] = y_coord1 - x_pixel*sin(rot1 + rot) + y_pixel*cos(rot1 + rot)

            hist += np.histogram2d(all_x_coords, all_y_coords, bins=[x_edges, y_edges])[0]

        total_hits = sum(map(sum, hist))
        print('total hits:', total_hits, num_detectors*num_ts)
        print('shape:', np.shape(hist), '<->', len(x_edges), len(y_edges))

        return hist
        
        """total_rot = np.radians(rot_angle[:last_sample] + rot)
        for i, x_off, y_off in zip(range(0, length), x_pixel, y_pixel):
            all_x_coords[i*last_sample: (i+1)*last_sample] = x_coord[:last_sample] + x_off*np.cos(total_rot) + y_off*np.sin(total_rot)
            all_y_coords[i*last_sample: (i+1)*last_sample] = y_coord[:last_sample] - x_off*np.sin(total_rot) + y_off*np.cos(total_rot)"""

    def hitmap(self, **kwargs):
        # FIXME cleanup, documentation, and optimization 

        max_acc = kwargs.get('max_acc', None)
        
        # remove points with high acceleration 
        if max_acc is None:
            x_coord = self.data['x_coord'].to_numpy()
            y_coord = self.data['y_coord'].to_numpy()
            rot_angle = self.data['rot_angle'].to_numpy()

            x_coord_rem = np.array([])
            y_coord_rem = np.array([])
            rot_angle_rem = np.array([])
        else:
            max_acc = u.Quantity(max_acc, u.deg/u.s).value
            total_acc = np.sqrt(self.data['x_acc']**2 + self.data['y_acc']**2)
            mask = total_acc < max_acc

            x_coord = self.data.loc[mask, 'x_coord'].to_numpy()
            y_coord = self.data.loc[mask, 'y_coord'].to_numpy()
            rot_angle = self.data.loc[mask, 'rot_angle'].to_numpy()

            x_coord_rem = self.data.loc[~mask, 'x_coord'].to_numpy()
            y_coord_rem = self.data.loc[~mask, 'y_coord'].to_numpy()
            rot_angle_rem = self.data.loc[~mask, 'rot_angle'].to_numpy()

        rot = u.Quantity(kwargs.get('rot', 0), u.deg).value
        plate_scale = u.Quantity(kwargs.get('plate_scale', 52*u.arcsec), u.deg).value
        pixel_size = u.Quantity(kwargs.get('pixel_size', 10*u.arcsec), u.deg).value

        ROOT = os.path.abspath(os.path.dirname(__file__))
        PIXELPOS_FILES = ['pixelpos1.txt', 'pixelpos2.txt', 'pixelpos3.txt']
        PIXELPOS_FILES = [os.path.join(ROOT, 'data', f) for f in PIXELPOS_FILES]
        
        # get pixel positions (convert meters->deg)
        x_pixel = np.array([])
        y_pixel = np.array([])

        for f in PIXELPOS_FILES:
            x, y = np.loadtxt(f, unpack=True)
            x_pixel = np.append(x_pixel, x)
            y_pixel = np.append(y_pixel, y)

        dist_btwn_detectors = math.sqrt((x_pixel[0] - x_pixel[1])**2 + (y_pixel[0] - y_pixel[1])**2)
        print('pixel_size =', dist_btwn_detectors)
        x_pixel = x_pixel/dist_btwn_detectors*plate_scale 
        y_pixel = y_pixel/dist_btwn_detectors*plate_scale 
        
        # define bin edges
        x_max = 2 #FIXME '
        y_max = 2
        x_edges = np.arange(-x_max, x_max+pixel_size, pixel_size)
        y_edges = np.arange(-y_max, y_max+pixel_size, pixel_size)
        print('x max min =', x_edges[0], x_edges[-1])
        print('y max min =', y_edges[0], y_edges[-1])

        hitmap_params = {
            'x_edges': x_edges, 'y_edges': y_edges,
            'x_pixel': x_pixel, 'y_pixel': y_pixel,
            'rot': rot
        }
        hist = self._generate_hitmap(x_coord, y_coord, rot_angle, **hitmap_params)
        hist_rem = self._generate_hitmap(x_coord_rem, y_coord_rem, rot_angle_rem, **hitmap_params)
        return

        # -- PLOTTING --

        fig = plt.figure(1)
        vmax1 = kwargs.get('vmax1')
        vmax2 = kwargs.get('vmax2')
        vmax3 = kwargs.get('vmax3')

        # plot histogram (kept)
        ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=3)
        pcm = ax1.imshow(hist.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], vmin=0, vmax=vmax1, interpolation='nearest', origin='lower')
        ax1.set_aspect('equal', 'box')
        ax1.set(xlabel='x offset (deg)', ylabel='y offset (deg)')
        ax1.set_title('Kept hits per pixel')

        if self.default_folder == 'curvy_pong':
            field = patches.Rectangle((-self.params['width']/2, -self.params['height']/2), width=self.params['width'], height=self.params['height'], linewidth=1, edgecolor='r', facecolor='none') 
            ax1.add_patch(field)
        ax1.axvline(x=0, c='black')
        ax1.axhline(y=0, c='black')

        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("bottom", size="3%", pad=0.5)
        fig.colorbar(pcm, cax=cax1, orientation='horizontal')

        # plot histogram (removed)
        ax2 = plt.subplot2grid((4, 4), (0, 1), rowspan=3)
        pcm = ax2.imshow(hist_rem.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], vmin=0, vmax=vmax2, interpolation='nearest', origin='lower')
        ax2.set_aspect('equal', 'box')
        ax2.set(xlabel='x offset (deg)', ylabel='y offset (deg)')
        ax2.set_title('Removed hits per pixel')

        if self.default_folder == 'curvy_pong':
            field = patches.Rectangle((-self.params['width']/2, -self.params['height']/2), width=self.params['width'], height=self.params['height'], linewidth=1, edgecolor='r', facecolor='none') 
            ax2.add_patch(field)
        ax2.axvline(x=0, c='black')
        ax2.axhline(y=0, c='black')

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("bottom", size="3%", pad=0.5)
        fig.colorbar(pcm, cax=cax2, orientation='horizontal')

        # plot histogram (combined)
        ax3 = plt.subplot2grid((4, 4), (0, 2), rowspan=3)
        pcm = ax3.imshow((hist + hist_rem).T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], vmin=0, vmax=vmax1, interpolation='nearest', origin='lower')
        ax3.set_aspect('equal', 'box')
        ax3.set(xlabel='x offset (deg)', ylabel='y offset (deg)')
        ax3.set_title('Total hits per pixel')

        if self.default_folder == 'curvy_pong':
            field = patches.Rectangle((-self.params['width']/2, -self.params['height']/2), width=self.params['width'], height=self.params['height'], linewidth=1, edgecolor='r', facecolor='none') 
            ax3.add_patch(field)
        ax3.scatter(self.data['x_coord'], self.data['y_coord'], color='black', s=0.001)
        ax3.axvline(x=0, c='black')
        ax3.axhline(y=0, c='black')

        divider2 = make_axes_locatable(ax3)
        cax3 = divider2.append_axes("bottom", size="3%", pad=0.5)
        fig.colorbar(pcm, cax=cax3, orientation='horizontal')

        kept_hits = sum(map(sum, hist))
        removed_hits = sum(map(sum, hist_rem))
        print(f'{removed_hits}/{kept_hits + removed_hits}')
        textstr = f'{round(removed_hits/(kept_hits+removed_hits)*100, 2)}% hits lost'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax3.text(0.1, 0.9, textstr, transform=ax3.transAxes, bbox=props)

        # plot detector pixel positions
        rot = u.Quantity(kwargs.get('rot', 0), u.rad).value
        ax6 = plt.subplot2grid((4, 4), (0, 3), rowspan=3, sharex=ax1, sharey=ax1)
        ax6.scatter(x_pixel*cos(rot) + y_pixel*sin(rot), -x_pixel*sin(rot) + y_pixel*cos(rot), s=0.01)
        ax6.set_aspect('equal', 'box')
        ax6.set(xlabel='x offset (deg)', ylabel='y offset (deg)')
        ax6.set_title('Detector Pixel Positions')
        divider6 = make_axes_locatable(ax6)
        cax6 = divider6.append_axes("bottom", size="3%", pad=0.5)
        cax6.axis('off')

        # bin line plot (#1)
        pixel_scale = kwargs.get('pixel_scale').to(u.deg).value

        ax4 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
        bin_index = round(x_edges[-1]/pixel_scale)
        bin_edge = x_edges[bin_index]
        y_values = hist[bin_index]
        y_values_rem = hist_rem[bin_index]

        ax4.plot(y_edges[:-1], y_values, label='Kept hits', drawstyle='steps')
        ax4.plot(y_edges[:-1], y_values_rem, label='Removed hits', drawstyle='steps')
        ax4.plot(y_edges[:-1], y_values + y_values_rem, label='Total hits', drawstyle='steps', color='black')

        if self.default_folder == 'curvy_pong':
            ax4.axvline(x=-self.params['width']/2, c='r')
            ax4.axvline(x=self.params['width']/2, c='r') 
        ax4.set(ylabel='Hits/Pixel', xlabel='y offset (deg)', ylim=(0, vmax3))
        ax4.set_title(f'Hit count in x={round(bin_edge, 5)} to x={round(bin_edge+pixel_scale, 5)} bin', fontsize=12)
        ax4.legend(loc='upper right')

        # bin line plot (#2)
        ax5 = plt.subplot2grid((4, 4), (3, 2), colspan=2)
        bin_index = round(x_edges[-1]/pixel_scale/2)
        bin_edge = x_edges[bin_index]
        y_values = hist[bin_index]
        y_values_rem = hist_rem[bin_index]

        ax5.plot(y_edges[:-1], y_values, label='Kept hits', drawstyle='steps')
        ax5.plot(y_edges[:-1], y_values_rem, label='Removed hits', drawstyle='steps')
        ax5.plot(y_edges[:-1], y_values + y_values_rem, label='Total hits', drawstyle='steps', color='black')

        if self.default_folder == 'curvy_pong':
            ax5.axvline(x=-self.params['width']/2, c='r')
            ax5.axvline(x=self.params['width']/2, c='r') 
        ax5.set(ylabel='Hits/Pixel', xlabel='y offset (deg)', ylim=(0, vmax3))
        ax5.set_title(f'Hit count in x={round(bin_edge, 5)} to x={round(bin_edge+pixel_scale, 5)} bin', fontsize=12)
        ax5.legend(loc='upper right')

        fig.tight_layout()
        plt.show()

    def plot(self, graphs=['coord', 'coord-time', 'vel', 'acc', 'jerk']):
        setting = None
        
        if 'coord' in graphs:
            fig_coord, ax_coord = plt.subplots(1, 2)
            ax_coord[0].plot(self.data['x_coord'], self.data['y_coord'])
            ax_coord[0].set_aspect('equal', 'box')
            ax_coord[0].set(xlabel='Right Ascension (degrees)', ylabel='Declination (degrees)', title='RA/DEC')
            ax_coord[0].grid()

            if 'az_coord' in self.data.columns and 'alt_coord' in self.data.columns:
                ax_coord[1].plot(self.data['az_coord'], self.data['alt_coord'])
                ax_coord[1].set_aspect('equal', 'box')
                ax_coord[1].set(xlabel='Azimuth (degrees)', ylabel='Altitude (degrees)', title=f'AZ/ALT {setting}')
                ax_coord[1].grid()

            fig_coord.tight_layout()

        if 'coord-time' in graphs:
            fig_coord_time, ax_coord_time = plt.subplots(2, 1, sharex=True, sharey=True)

            ax_coord_time[0].plot(self.data['time_offset'], self.data['x_coord'], label='RA')
            ax_coord_time[0].plot(self.data['time_offset'], self.data['y_coord'], label='DEC')
            ax_coord_time[0].legend(loc='upper right')
            ax_coord_time[0].set(xlabel='time offset (s)', ylabel='Position offset (deg)', title='RA/DEC Position')
            ax_coord_time[0].grid()

            if 'az_coord' in self.data.columns and 'alt_coord' in self.data.columns:
                ax_coord_time[1].plot(self.data['time_offset'], (self.data['az_coord'] - self.data.loc[0, 'az_coord'])*cos(pi/6), label='AZ')
                ax_coord_time[1].plot(self.data['time_offset'], self.data['alt_coord'] - self.data.loc[0, 'alt_coord'], label='ALT')
                ax_coord_time[1].legend(loc='upper right')
                ax_coord_time[1].set(xlabel='time offset (s)', ylabel='Position offset (deg)', title=f'AZ/ALT {setting}')
                ax_coord_time[1].grid()

            fig_coord_time.tight_layout()
        
        if 'vel' in graphs:
            fig_vel, ax_vel = plt.subplots(2, 1, sharex=True, sharey=True)

            total_vel = np.sqrt(self.data['x_vel']**2 + self.data['y_vel']**2)
            ax_vel[0].plot(self.data['time_offset'], total_vel, label='Total', c='black', ls='dashed', alpha=0.25)
            ax_vel[0].plot(self.data['time_offset'], self.data['x_vel'], label='RA')
            ax_vel[0].plot(self.data['time_offset'], self.data['y_vel'], label='DEC')
            ax_vel[0].legend(loc='upper right')
            ax_vel[0].set(xlabel='time offset (s)', ylabel='velocity (deg/s)', title='RA/DEC Velocity')
            ax_vel[0].grid()

            if 'az_vel' in self.data.columns and 'alt_vel' in self.data.columns:
                total_vel = np.sqrt(self.data['az_vel']**2 + self.data['alt_vel']**2)
                ax_vel[1].plot(self.data['time_offset'], total_vel, label='Total', c='black', ls='dashed', alpha=0.25)
                ax_vel[1].plot(self.data['time_offset'], self.data['az_vel'], label='AZ')
                ax_vel[1].plot(self.data['time_offset'], self.data['alt_vel'], label='ALT')
                ax_vel[1].legend(loc='upper right')
                ax_vel[1].set(xlabel='time offset (s)', ylabel='velocity (deg/s)', title=f'AZ/ALT {setting}')
                ax_vel[1].grid()

            fig_vel.tight_layout()
        
        if 'acc' in graphs:
            fig_acc, ax_acc = plt.subplots(2, 1, sharex=True, sharey=True)

            total_acc = np.sqrt(self.data['x_acc']**2 + self.data['y_acc']**2)
            ax_acc[0].plot(self.data['time_offset'], total_acc, label='Total', c='black', ls='dashed', alpha=0.25)
            ax_acc[0].plot(self.data['time_offset'], self.data['x_acc'], label='RA')
            ax_acc[0].plot(self.data['time_offset'], self.data['y_acc'], label='DEC')
            ax_acc[0].legend(loc='upper right')
            ax_acc[0].set(xlabel='time offset (s)', ylabel='acceleration (deg/s^2)', title='RA/DEC Acceleration')
            ax_acc[0].grid()

            if 'az_acc' in self.data.columns and 'alt_acc' in self.data.columns:
                total_acc = np.sqrt(self.data['az_acc']**2 + self.data['alt_acc']**2)
                ax_acc[1].plot(self.data['time_offset'], total_acc, label='Total', c='black', ls='dashed', alpha=0.25)
                ax_acc[1].plot(self.data['time_offset'], self.data['az_acc'], label='AZ')
                ax_acc[1].plot(self.data['time_offset'], self.data['alt_acc'], label='ALT')
                ax_acc[1].legend(loc='upper right')
                ax_acc[1].set(xlabel='time offset (s)', ylabel='acceleration (deg/s^2)', title=f'AZ/ALT {setting}')
                ax_acc[1].grid()

            fig_acc.tight_layout()

        if 'jerk' in graphs:
            fig_jerk, ax_jerk = plt.subplots(2, 1, sharex=True, sharey=True)

            total_jerk = np.sqrt(self.data['x_jerk']**2 + self.data['y_jerk']**2)
            ax_jerk[0].plot(self.data['time_offset'], self.data['x_jerk'], label='RA')
            ax_jerk[0].plot(self.data['time_offset'], self.data['y_jerk'], label='DEC')
            ax_jerk[0].plot(self.data['time_offset'], total_jerk, label='Total', c='black', ls='dashed', alpha=0.25)
            ax_jerk[0].legend(loc='upper right')
            ax_jerk[0].set(xlabel='time offset (s)', ylabel='Jerk (deg/s^2)', title='RA/DEC Jerk')
            ax_jerk[0].grid()

            if 'az_jerk' in self.data.columns and 'alt_jerk' in self.data.columns:
                total_jerk = np.sqrt(self.data['az_jerk']**2 + self.data['alt_jerk']**2)
                ax_jerk[1].plot(self.data['time_offset'], self.data['az_jerk'], label='AZ')
                ax_jerk[1].plot(self.data['time_offset'], self.data['alt_jerk'], label='ALT')
                ax_jerk[1].plot(self.data['time_offset'], total_jerk, label='Total', c='black', ls='dashed', alpha=0.25)
                ax_jerk[1].legend(loc='upper right')
                ax_jerk[1].set(xlabel='time offset (s)', ylabel='Jerk (deg/s^2)', title=f'AZ/ALT {setting})')
                ax_jerk[1].grid()

            fig_jerk.tight_layout()
        
        if 'quiver' in graphs:
            fig_quiver, ax_quiver = plt.subplots(1, 2, sharex=True, sharey=True)
            subsample = 50
            endpoint = None

            # --- ACCELERATION ---

            # plot acc
            total_acc = np.sqrt(self.data['x_acc']**2 + self.data['y_acc']**2).to_numpy()
            ax_quiver[0].plot(self.data['x_coord'], self.data['y_coord'], alpha=0.25, color='black')
            pcm = ax_quiver[0].quiver(
                self.data['x_coord'].to_numpy()[:endpoint:subsample], self.data['y_coord'].to_numpy()[:endpoint:subsample], 
                self.data['x_acc'].to_numpy()[:endpoint:subsample], self.data['y_acc'].to_numpy()[:endpoint:subsample],
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
            self.data['x_jerk'] = self._central_diff(self.data['x_acc'].to_numpy(), self.sample_interval)
            self.data['y_jerk'] = self._central_diff(self.data['y_acc'].to_numpy(), self.sample_interval)
            total_jerk = np.sqrt(self.data['x_jerk']**2 + self.data['y_jerk']**2).to_numpy()

            # plot jerk
            ax_quiver[1].plot(self.data['x_coord'], self.data['y_coord'], alpha=0.25, color='black')
            pcm = ax_quiver[1].quiver(
                self.data['x_coord'].to_numpy()[:endpoint:subsample], self.data['y_coord'].to_numpy()[:endpoint:subsample], 
                self.data['x_jerk'].to_numpy()[:endpoint:subsample], self.data['y_jerk'].to_numpy()[:endpoint:subsample]/3600,
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

class CurvyPong(ScanPattern):
    """
    The Curvy Pong pattern allows for an approximation of a Pong pattern while avoiding 
    sharp turnarounds at the vertices. 
    
    See "The Impact of Scanning Pattern Strategies on Uniform Sky Coverage of Large Maps" 
    (SCUBA Project SC2/ANA/S210/008) for details of implementation. 

    Paramters
    -------------------------
    data_csv : string, optional
        File path to data file
    param_csv : string, optional, defaults to finding default parameter file
        File path to parameter file

    **kwargs 
    num_terms : int, optional
        Number of terms in the triangle wave expansion 
    width, height : angle-like, optional
        Width and height of field of view
        If no units specified, defaults to deg
    spacing : angle-like, optional
        Space between adjacent (parallel) scan lines in the Pong pattern
        If no units specified, defaults to deg
    velocity : angle/time-like, optional
        Target magnitude of the scan velocity excluding turn-arounds
        If no units specified, defaults to deg/s
    angle : angle-like, optional, defaults to 0 deg
        Position angle of the box in the native coordinate system
        If no units specified, defaults to deg
    sample_interval : time, optional, defaults to 400 Hz
        Time between read-outs 
        If no units specified, defaults to s
    """

    default_folder = 'curvy_pong'
    default_param_csv = 'curvy_pong_params.csv'

    def __init__(self, data_csv=None, param_csv=None, **kwargs):
        
        # pass by csv file
        if not data_csv is None:
            super().__init__(data_csv, param_csv)

        # initialize a new scan 
        else:
            num_terms = kwargs.pop('num_terms')
            width = u.Quantity(kwargs.pop('width'), u.deg).value
            height = u.Quantity(kwargs.pop('height'), u.deg).value
            spacing = u.Quantity(kwargs.pop('spacing'), u.deg).value
            velocity = u.Quantity(kwargs.pop('velocity'), u.deg/u.s).value
            angle = u.Quantity(kwargs.pop('angle', 0), u.deg).value
            sample_interval = u.Quantity(kwargs.pop('sample_interval', 0.0025), u.s).value

            self.params = {
                'num_terms': num_terms,
                'width': width, 'height': height, 'spacing': spacing, 'velocity': velocity,
                'angle': angle, 'sample_interval': sample_interval
            }

            self.data = self._generate_scan(num_terms, width, height, spacing, velocity, angle, sample_interval)
    
    def _fourier_expansion(self, num_terms, amp, t_count, peri):
        N = num_terms*2 - 1
        a = (8*amp)/(pi**2)
        b = 2*pi/peri

        position = 0
        velocity = 0
        acc = 0
        jerk = 0
        for n in range(1, N+1, 2):
            c = math.pow(-1, (n-1)/2)/n**2 
            position += c * sin(b*n*t_count)
            velocity += c*n * cos(b*n*t_count)
            acc      += c*n**2 * sin(b*n*t_count)
            jerk     += c*n**3 * cos(b*n*t_count)

        position *= a
        velocity *= a*b
        acc      *= -a*b**2
        jerk     *= -a*b**3
        return position, velocity, acc, jerk

    def _generate_scan(self, num_terms, width, height, spacing, velocity, angle, sample_interval):

        # Determine number of vertices (reflection points) along each side of the
        # box which satisfies the common-factors criterion and the requested size / spacing    

        vert_spacing = sqrt(2)*spacing
        x_numvert = ceil(width/vert_spacing)
        y_numvert = ceil(height/vert_spacing)
 
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

        pongcount = ceil(period/sample_interval)
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
        x_jerk = []
        y_jerk = []

        for i in range(pongcount + 1):
            tx, ttx, tttx, jerkx = self._fourier_expansion(num_terms, amp_x, t_count, peri_x)
            ty, tty, ttty, jerky = self._fourier_expansion(num_terms, amp_y, t_count, peri_y)

            x_coord.append(tx*cos(angle) - ty*sin(angle))
            y_coord.append(tx*sin(angle) + ty*cos(angle))
            x_vel.append(ttx*cos(angle) - tty*sin(angle))
            y_vel.append(ttx*sin(angle) + tty*cos(angle))
            x_acc.append(tttx*cos(angle) - ttty*sin(angle))
            y_acc.append(tttx*sin(angle) + ttty*cos(angle))
            x_jerk.append(jerkx*cos(angle) - jerky*sin(angle))
            y_jerk.append(jerkx*sin(angle) + jerky*cos(angle))

            time_offset.append(t_count)
            t_count += sample_interval
        
        return pd.DataFrame({
            'time_offset': time_offset, 
            'x_coord': np.array(x_coord), 'y_coord': np.array(y_coord), 
            'x_vel': x_vel, 'y_vel': y_vel,
            'x_acc': x_acc, 'y_acc': y_acc,
            'x_jerk': x_jerk, 'y_jerk': y_jerk
        })

class Daisy(ScanPattern):
    """
    See "CV Daisy - JCMT small area scanning patter" (JCMT TCS/UN/005) for details of implementation. 

    Parameters
    -----------------------------------
    speed : angle-like/time-like
        Constant velocity (CV) for scan to go at. 
    acc : angle-like/time-like^2
        Acceleration at start of pattern
    R0 : angle-like
        Radius R0
    Rt : angle-like
        Turn radius
    Ra : angle-like
        Avoidance radius
    T : time-like
        Total time of the simulation
    sample_interval : time-like [default 400 Hz]
        Time step
    y : angle-like
        start offset in y [default 0"]
    """

    default_folder = 'daisy'
    default_param_csv = 'daisy_params'

    def __init__(self, data_csv=None, param_csv=None, **kwargs):
        
        # pass by csv file
        if not data_csv is None:
            super().__init__(data_csv, param_csv)
        
        # initialize new scan
        else:
            speed = u.Quantity(kwargs['speed'], u.arcsec/u.s).value
            acc = u.Quantity(kwargs['acc'], u.arcsec/u.s/u.s).value
            R0 = u.Quantity(kwargs['R0'], u.arcsec).value
            Rt = u.Quantity(kwargs['Rt'], u.arcsec).value
            Ra = u.Quantity(kwargs['Ra'], u.arcsec).value
            T = u.Quantity(kwargs['T'], u.s).value
            sample_interval = u.Quantity(kwargs.get('sample_interval', 1/400*u.s), u.s).value
            y = u.Quantity(kwargs.get('y', 0*u.arcsec), u.arcsec).value

            self.params = {
                'speed': speed, 'acc': acc,
                'R0': R0, 'Rt': Rt, 'Ra': Ra,
                'T': T, 'sample_interval': sample_interval,
                'y': y
            }

            self.data = self._generate_scan(speed, acc, R0, Rt, Ra, T, sample_interval, y)
        
    def _generate_scan(self, speed, acc, R0, Rt, Ra, T, dt, y):

        xval = [] # x,y arrays for plotting 
        yval = [] 
        x_vel = [] # speed array for plotting 
        y_vel = []

        (vx, vy) = (1.0, 0.0) # Tangent vector & start value
        (x, y) = (0.0, y) # Position vector & start value
        N = int(T/dt) + 1 # number of steps 
        R1 = min(R0, Ra) # Effective avoidance radius so Ra is not used if Ra > R0 

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
            xval.append(x)
            yval.append(y)
            x_vel.append(speed*vx)
            y_vel.append(speed*vy)

        # Compute arrays for plotting 
        """x_vel = self._central_diff(xval)
        x_acc = self._central_diff(x_vel)
        x_jerk = self._central_diff(x_acc)

        y_vel = self._central_diff(yval)
        y_acc = self._central_diff(y_vel)
        y_jerk = self._central_diff(y_acc)"""

        xval = np.array(xval)
        yval = np.array(yval)
        x_vel = np.array(x_vel)
        y_vel = np.array(y_vel)

        ax = -2*xval[1: -1] + xval[0:-2] + xval[2:] # numerical acc in x 
        ay = -2*yval[1: -1] + yval[0:-2] + yval[2:] # numerical acc in y 
        x_acc = np.append(np.array([0]), ax/dt/dt)
        y_acc = np.append(np.array([0]), ay/dt/dt)
        x_acc = np.append(x_acc, 0)
        y_acc = np.append(y_acc, 0)
        x_jerk = self._central_diff(x_acc)
        y_jerk = self._central_diff(y_acc)

        return pd.DataFrame({
            'time_offset': np.arange(0, T, dt), 
            'x_coord': xval/3600, 'y_coord': yval/3600, 
            'x_vel': x_vel/3600, 'y_vel': y_vel/3600,
            'x_acc': x_acc/3600, 'y_acc': y_acc/3600,
            'x_jerk': x_jerk/3600, 'y_jerk': y_jerk/3600
        })


# ------------------------
# PLOTTING FUNCTIONS
# ------------------------

def compare_num_terms(first=2, last=5, **kwargs): # FIXME from param_csv or data_csv, pick which plots
    num_subplots = last-first + 1
    num_col = 2
    num_row = ceil(num_subplots/num_col)

    """fig_coord, ax_coord = plt.subplots(num_row, num_col, sharex=True, sharey=True)
    fig_xcoord, ax_xcoord = plt.subplots(num_row, num_col, sharex=True, sharey=True)
    fig_xvel, ax_xvel = plt.subplots(num_row, num_col, sharex=True, sharey=True)
    fig_xacc, ax_xacc = plt.subplots(num_row, num_col, sharex=True, sharey=True)
    fig_xjerk, ax_xjerk = plt.subplots(num_row, num_col, sharex=True, sharey=True)"""

    for N in range(first, last+1):
        kwargs['num_terms'] = N
        scan = CurvyPong(**kwargs)

        row = math.floor((N-2)/2)
        col = N%2

        scan.set_setting(ra=0, dec=0, alt=30, date='2001-12-09')
        scan.hitmap(**kwargs)

        """ax_coord[row, col].plot(scan.data['x_coord'], scan.data['y_coord'])
        ax_coord[row, col].set_title(f'# of terms = {N} in expansion')
        ax_coord[row, col].set(xlabel='x offset [deg]', ylabel='y offset [deg]')
        ax_coord[row, col].set_aspect('equal', 'box')
        ax_coord[row, col].grid()

        ax_xcoord[row, col].plot(scan.data['time_offset'], scan.data['x_coord'])
        ax_xcoord[row, col].set_title(f'# of terms = {N} in expansion')
        ax_xcoord[row, col].set(xlabel='time offset [s]', ylabel='x offset [deg]')
        ax_xcoord[row, col].grid()

        ax_xvel[row, col].plot(scan.data['time_offset'], scan.data['x_vel'])
        ax_xvel[row, col].set_title(f'# of terms = {N} in expansion')
        ax_xvel[row, col].set(xlabel='time offset [s]', ylabel='x velocity [deg/s]')
        ax_xvel[row, col].grid()

        ax_xacc[row, col].plot(scan.data['time_offset'], scan.data['x_acc'])
        ax_xacc[row, col].set_title(f'# of terms = {N} in expansion')
        ax_xacc[row, col].set(xlabel='time offset [s]', ylabel='x acceleration [deg/s^2]')
        ax_xacc[row, col].grid()

        ax_xjerk[row, col].plot(scan.data['time_offset'], scan.data['x_jerk'])
        ax_xjerk[row, col].set_title(f'# of terms = {N} in expansion')
        ax_xjerk[row, col].set(xlabel='time offset [s]', ylabel='x jerk [deg/s]')
        ax_xjerk[row, col].grid()"""


    """fig_coord.tight_layout()
    fig_xcoord.tight_layout()
    fig_xvel.tight_layout()
    fig_xacc.tight_layout()
    fig_xjerk.tight_layout()"""

    #plt.show()


# ------------------------

if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()

    """scan = CurvyPong(num_terms=5, width=2, height=7000*u.arcsec, spacing='500 arcsec', velocity='1000 arcsec/s', sample_interval=0.2)
    #scan = CurvyPong('ra0dec0alt30_20011209.csv')
    """

    """compare_num_terms(
        plate_scale=52*u.arcsec, pixel_scale=10*u.arcsec, max_acc=0.2*u.deg/u.s, 
        width=2*u.deg, height=2*u.deg, spacing=500*u.arcsec, velocity=1/3/sqrt(2)*u.deg/u.s, sample_interval=0.0025*u.s,
        vmax1=900,vmax2=350,vmax3=900
    )"""

    #scan = Daisy(speed=1000*u.arcsec/u.s, acc=0.2*u.deg/u.s/u.s, R0=1*u.deg, Rt=1*u.deg, Ra=1/4*u.deg, T=720*u.s, sample_interval=1/40*u.s)
    #scan.set_setting(ra=0, dec=0, alt=30, date='2001-12-09')
    #scan.to_csv('daisy_less.csv')
    scan = Daisy('daisy_less.csv')
    #scan.plot()
    scan.hitmap(plate_scale=52*u.arcsec, pixel_scale=10*u.arcsec, max_acc=None)

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats()
    with open('delete/1.txt', 'w+') as f:
        f.write(s.getvalue())

