import os
import warnings 
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

    def hitmap(self, **kwargs):
        current_path = os.path.dirname(os.path.abspath(__file__))
        pixelpos_files = ['pixelpos1.txt', 'pixelpos2.txt', 'pixelpos3.txt']
        pixelpos_files = [os.path.join(current_path, f) for f in pixelpos_files]
        percent = u.Quantity(kwargs.get('percent', 1))

        rot = u.Quantity(kwargs.get('rot', 0), u.deg).value
        max_acc = kwargs.get('max_acc', None)
        plate_scale = u.Quantity(kwargs.get('plate_scale', 26*u.arcsec), u.arcsec).value
        pixel_size = int(u.Quantity(kwargs.get('pixel_size', 10*u.arcsec), u.arcsec).value)
        
        # remove points with high acceleration 
        total_azalt_acc = np.sqrt(self.data['az_acc']**2 + self.data['alt_acc']**2)

        if max_acc is None:
            x_coord = self.data['x_coord'].to_numpy()*3600
            y_coord = self.data['y_coord'].to_numpy()*3600
            rot_angle = self.data['rot_angle'].to_numpy()
        else:
            max_acc = u.Quantity(max_acc, u.deg/u.s).value
            mask = total_azalt_acc < max_acc
            x_coord = self.data.loc[mask, 'x_coord'].to_numpy()*3600
            y_coord = self.data.loc[mask, 'y_coord'].to_numpy()*3600
            rot_angle = self.data.loc[mask, 'rot_angle'].to_numpy()
        
        # get pixel positions (convert meters->deg)
        x_pixel = np.array([])
        y_pixel = np.array([])

        for f in pixelpos_files:
            x, y = np.loadtxt(f, unpack=True)
            x_pixel = np.append(x_pixel, x)
            y_pixel = np.append(y_pixel, y)

        dist_btwn_detectors = math.sqrt((x_pixel[0] - x_pixel[1])**2 + (y_pixel[0] - y_pixel[1])**2)
        length = len(x_pixel)
        print('pixel_size =', dist_btwn_detectors)
        print('total number of detector pixels =', len(x_pixel))
        x_pixel = x_pixel/dist_btwn_detectors*plate_scale 
        y_pixel = y_pixel/dist_btwn_detectors*plate_scale 
        
        # define bin edges
        x_max = 5400 #FIXME '
        y_max = 5400
        x_edges = range(-x_max, x_max+pixel_size, int(pixel_size))
        y_edges = range(-y_max, y_max+pixel_size, int(pixel_size))
        print('x max min =', x_edges[0], x_edges[-1])
        print('y max min =', y_edges[0], y_edges[-1])

        # get only first x percent of scan
        last_sample = ceil(len(x_coord)*percent)

        # sort all positions with individual detector offset into a 2D histogram
        all_x_coords = np.zeros(last_sample * length)
        all_y_coords = np.zeros(last_sample * length)
        rot = radians(rot)


        for i, x_coord1, y_coord1, rot1 in zip(range(0, last_sample), x_coord, y_coord, np.radians(rot_angle[:last_sample])):
            all_x_coords[i*length: (i+1)*length] = x_coord1 + x_pixel*np.cos(rot1 + rot) + y_pixel*np.sin(rot1 + rot)
            all_y_coords[i*length: (i+1)*length] = y_coord1 - x_pixel*np.sin(rot1 + rot) + y_pixel*np.cos(rot1 + rot)
        
        total_rot = np.radians(rot_angle[:last_sample] + rot)
        for i, x_off, y_off in zip(range(0, length), x_pixel, y_pixel):
            all_x_coords[i*last_sample: (i+1)*last_sample] = x_coord[:last_sample] + x_off*np.cos(total_rot) + y_off*np.sin(total_rot)
            all_y_coords[i*last_sample: (i+1)*last_sample] = y_coord[:last_sample] - x_off*np.sin(total_rot) + y_off*np.cos(total_rot)

        """hist = np.zeros( (len(x_edges)-1 , len(y_edges)-1) )
        total_rot = np.radians(rot_angle[:last_sample] + rot)
        for x_off, y_off in zip(x_pixel, y_pixel):
            x_off_rot = x_off*np.cos(total_rot) + y_off*np.sin(total_rot)
            y_off_rot = -x_off*np.sin(total_rot) + y_off*np.cos(total_rot)
            hist += np.histogram2d(x_coord[:last_sample] + x_off_rot, y_coord[:last_sample] + y_off_rot, bins=[x_edges, y_edges])[0]"""

        hist = np.histogram2d(all_x_coords, all_y_coords, bins=[x_edges, y_edges])[0]
        print('shape:', np.shape(hist), '<->', len(x_edges), len(y_edges))

        # -- PLOTTING --

        fig = plt.figure(1)

        # plot histogram
        ax1 = plt.subplot2grid((3, 3), (0, 1), rowspan=2)
        pcm = ax1.imshow(hist.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], vmin=0, interpolation='nearest', origin='lower')
        ax1.set_aspect('equal', 'box')
        field = patches.Rectangle((-self.params['width']*3600/2, -self.params['height']*3600/2), width=self.params['width']*3600, height=self.params['height']*3600, linewidth=1, edgecolor='r', facecolor='none') 
        ax1.add_patch(field)
        subtitle = f'alt=30, max acc={max_acc}, pixel size={round(pixel_size, 5)}'
        ax1.set(xlabel='x offset (deg)', ylabel='y offset (deg)')
        ax1.set_title('Kept hits per pixel\n'+subtitle, fontsize=12)
        ax1.scatter(x_coord[:last_sample], y_coord[:last_sample], color='r', s=0.001)
        ax1.axvline(x=0, c='black')
        ax1.axhline(y=0, c='black')
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("bottom", size="3%", pad=0.5)
        fig.colorbar(pcm, cax=cax1, orientation='horizontal')

        fig.tight_layout()
        #plt.show()


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
    sample_interval : time, optional, defaults to 0.002 s
        Time between read-outs 
        If no units specified, defaults to s
    """

    default_folder = 'curvy_pong'
    default_param_csv = 'curvy_pong_params.csv'

    def __init__(self, data_csv=None, param_csv=None, **kwargs):
        
        # pass by csv file
        if not data_csv is None:
            if bool(kwargs):
                raise ValueError(f'Additional arguments supplied when data_csv={data_csv} was already given.')
            super().__init__(data_csv, param_csv)

        # initialize a new scan 
        else:
            num_terms = kwargs.pop('num_terms')
            width = u.Quantity(kwargs.pop('width'), u.deg).value
            height = u.Quantity(kwargs.pop('height'), u.deg).value
            spacing = u.Quantity(kwargs.pop('spacing'), u.deg).value
            velocity = u.Quantity(kwargs.pop('velocity'), u.deg/u.s).value
            angle = u.Quantity(kwargs.pop('angle', 0), u.deg).value
            sample_interval = u.Quantity(kwargs.pop('sample_interval', 0.002), u.s).value

            if bool(kwargs):
                raise ValueError(f'Additional arguments supplied: {kwargs}')

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

    @staticmethod
    def generate_scan(self, num_terms, width, height, spacing, velocity, angle, sample_interval):

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




        
        

# ------------------------
# PLOTTING FUNCTIONS
# ------------------------



if __name__ == '__main__':
    #scan = CurvyPong(num_terms=5, width=2, height=7000*u.arcsec, spacing='500 arcsec', velocity='1000 arcsec/s', sample_interval=0.03)
    #scan.set_setting(ra=0, dec=0, alt=60, date='2001-12-09')
    #scan.to_csv('curvy_pong_less.csv')
    scan = CurvyPong('curvy_pong_less.csv')

    pr = cProfile.Profile()
    pr.enable()
    my_result = scan.hitmap(plate_scale=26*u.arcsec, pixel_scale=10*u.arcsec)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats()
    with open('del.txt', 'w+') as f:
        f.write(s.getvalue())
            


