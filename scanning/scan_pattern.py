import math
from math import pi, sin, cos, tan, sqrt
from astropy.coordinates.earth import EarthLocation
from astropy.utils.misc import isiterable
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

import warnings
import json

from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u

import matplotlib.pyplot as plt

from scanning import FYST_LOC, _central_diff, Instrument

"""
specify units when passing in x, y
"""

##################
#  SKY PATTERN 
##################

class SkyPattern():
    """
    Attributes
    ---------------------------
    data : pd.DataFrame : time_offset [s], x_coord [deg], y_coord [deg]
    param : dict
    repeatable
    """

    _param_unit = {'num_repeat': u.dimensionless_unscaled, 'sample_interval': u.s}
    _data_unit = {'time_offset': u.s, 'x_coord': u.deg, 'y_coord': u.deg}

    def __init__(self, data, repeatable=False, **kwargs) -> None:
        """
        Initialize an arbitrary scan. 

        Parameters
        --------------------------------
        data : str; ndarray, Iterable, dict, or DataFrame
            File path to csv file. Dict can contain Series, arrays, constants, dataclass or list-like objects. 
            Has columns 'time_offset', 'x_coord', 'y_coord'
        repeatable : bool, default False
            Whether this pattern can repeat itself (end where it starts).
        
        **kwargs
        max_scan_duration : time-like, default unit sec
            Maximum total scan time to determine number of repeats. Must be positive. 
            Cannot be used with num_repeat. 'repeatable' must be True.
        num_repeat : int, default 1 
            Number of repeats of the pattern. Must be >= 1
            Cannot be used with max_scan_duration. 'repeatable' must be True.
        """
        self.repeatable = repeatable 

        if isinstance(data, str):
            data = pd.read_csv(data, index_col=False, usecols=['time_offset', 'x_coord', 'y_coord'])
        else:
            data = pd.DataFrame(data, columns=['time_offset', 'x_coord', 'y_coord'])

        # determine sample_interval 
        sample_interval_list = np.diff(data['time_offset'].to_numpy())
        if np.std(sample_interval_list)/np.mean(sample_interval_list) <= 0.01:
            sample_interval = np.mean(sample_interval_list)
        else:
            raise ValueError('sample_interval must be constant')
        kwargs['sample_interval'] = sample_interval

        self.param = self._clean_param(**kwargs)
        self.data = self._repeat_scan(data)

    def _clean_param(self, **kwargs):
        kwarg_keys = kwargs.keys()

        # determining number of repeats 
        if self.repeatable:
            if 'max_scan_duration' in kwarg_keys and 'num_repeat' in kwarg_keys:
                raise ValueError('max_scan_duration and num_repeat cannot be inputted together')
            elif 'max_scan_duration' in kwarg_keys:
                kwargs['num_repeat'] = math.nan
                kwargs['max_scan_duration'] = u.Quantity(kwargs['max_scan_duration'], u.s).value
            else:
                kwargs['num_repeat'] = int(kwargs.get('num_repeat', 1))  

        # checking if repeats aren't passed if a non-repeatable pattern
        else:
            if 'max_scan_duration' in kwarg_keys or 'num_repeat' in kwarg_keys:
                raise ValueError('this is not a repeatable SkyPattern, so max_scan_duration and num_repeat cannot be initialized')

        return kwargs

    def _repeat_scan(self, data):
        one_scan_duration = data.iloc[-1]['time_offset'] + self.sample_interval.value

        # determine number of repeats
        num_repeat = self.num_repeat
        if math.isnan(num_repeat):
            max_scan_duration = self.param.pop('max_scan_duration') # only store number of repeats, not maximum scan duration
            num_repeat = math.floor(max_scan_duration/one_scan_duration)
            if num_repeat < 1:
                raise ValueError(f'max_scan_duration = {max_scan_duration} s is too short, one scan duration is {one_scan_duration} second')

            self.param['num_repeat'] = num_repeat

        # repeat pattern if necessary 
        time_offset = data['time_offset']
        if num_repeat > 1:
            data_temp = data.copy()
            for i in range(1, num_repeat):
                data_temp['time_offset'] = time_offset + one_scan_duration*i
                data = data.append(data_temp, ignore_index=True)

        return data

    def save_data(self, path_or_buf=None, columns='default', include_repeats=True):
        """
        Parameters
        ----------------------------
        path_or_buf : str or file handle, default None
            File path or object, if None is provided the result is returned as a dictionary.
        columns : sequence or str, default 'default'
            Columns to write. 
            'default' for ['time_offset', 'x_coord', 'y_coord']
            'all' for ['time_offset', 'x_coord', 'y_coord', 'x_vel', 'y_vel', 'vel', 'x_acc', 'y_acc', 'acc', 'x_jerk', 'y_jerk', 'jerk']
        include_repeats : bool, default 'True'
            include repeats of the SkyPattern
        """

        # replace str options
        if columns == 'default':
            columns = ['time_offset', 'x_coord', 'y_coord']
        elif columns == 'all':
            columns = ['time_offset', 'x_coord', 'y_coord', 'x_vel', 'y_vel', 'vel', 'x_acc', 'y_acc', 'acc', 'x_jerk', 'y_jerk', 'jerk']
        
        data = self.data.copy()

        # whether to include repetitions 
        if not include_repeats and self.num_repeat > 1:
            before_index = int(len(data.index)/self.num_repeat)
            data = data.iloc[:before_index]

        # calculate accelerations and jerks if necessary
        if 'x_vel' in columns:
            data['x_vel'] = self.x_vel.value
        if 'y_vel' in columns:
            data['y_vel'] = self.y_vel.value
        if 'vel' in columns:
            data['vel'] = self.vel.value

        if 'x_acc' in columns:
            data['x_acc'] = self.x_acc.value
        if 'y_acc' in columns:
            data['y_acc'] = self.y_acc.value
        if 'acc' in columns:
            data['acc'] = self.acc.value

        if 'x_jerk' in columns:
            data['x_jerk'] = self.x_jerk.value
        if 'y_jerk' in columns:
            data['y_jerk'] = self.y_jerk.value
        if 'jerk' in columns:
            data['jerk'] = self.jerk.value
        
        # save data file 
        if path_or_buf is None:
            return data[columns].to_dict()
        else:
            data.to_csv(path_or_buf, columns=columns, index=False)

    def save_param(self, path_or_buf=None):
        """
        Parameters
        ----------------------------
        path_or_buf : str or file handle, default None
            File path or object, if None is provided the result is returned as a dictionary.
        """
        
        param_temp = self.param.copy()

        # save param_json
        if path_or_buf is None:
            return param_temp
        else:
            with open(path_or_buf, 'w') as f:
                json.dump(param_temp, f)

    # GETTERS

    def __getattr__(self, attr):

        if attr in self.param.keys():
            if self._param_unit[attr] is u.dimensionless_unscaled:
                return self.param[attr]
            else:
                return self.param[attr]*self._param_unit[attr]
        elif attr in self.data.columns:
            return self.data[attr].to_numpy()*self._data_unit[attr]
        else:
            raise AttributeError(f'attribtue {attr} not found')

    @property
    def scan_duration(self):
        return self.time_offset[-1] + self.sample_interval

    @property
    def x_vel(self):
        return _central_diff(self.x_coord.value, self.sample_interval.value)*u.deg/u.s

    @property
    def y_vel(self):
        return _central_diff(self.y_coord.value, self.sample_interval.value)*u.deg/u.s

    @property
    def vel(self):
        return np.sqrt(self.x_vel**2 + self.y_vel**2)

    @property
    def x_acc(self):
        return _central_diff(self.x_vel.value, self.sample_interval.value)*u.deg/u.s/u.s

    @property
    def y_acc(self):
        return _central_diff(self.y_vel, self.sample_interval.value)*u.deg/u.s/u.s
    
    @property
    def acc(self):
        return np.sqrt(self.x_acc**2 + self.y_acc**2)

    @property
    def x_jerk(self):
        return _central_diff(self.x_acc.value, self.sample_interval.value)*u.deg/(u.s)**3
    
    @property
    def y_jerk(self):
        return _central_diff(self.y_acc.value, self.sample_interval.value)*u.deg/(u.s)**3
    
    @property
    def jerk(self):
        return np.sqrt(self.x_jerk**2 + self.y_jerk**2)

class Pong(SkyPattern):
    """
    The Curvy Pong pattern allows for an approximation of a Pong pattern while avoiding 
    sharp turnarounds at the vertices. 
    
    See "The Impact of Scanning Pattern Strategies on Uniform Sky Coverage of Large Maps" 
    (SCUBA Project SC2/ANA/S210/008) for details of implementation. 
    """

    _param_unit = {
        'num_term': u.dimensionless_unscaled,
        'width': u.deg, 'height': u.deg, 'spacing': u.deg,
        'velocity': u.deg/u.s, 'angle': u.deg, 'sample_interval': u.s, 
        'num_repeat': u.dimensionless_unscaled
    }
    repeatable = True

    def __init__(self, param_json=None, **kwargs) -> None:
        """
        Initialize a Pong pattern by passing a parameter file or dictionary and optionally 
        adding keywords to overwrite existing ones:
            option1 : Pong(param_json, **kwargs) 
        or building from scratch: 
            option2 : Pong(**kwargs)

        Parameters
        ---------------------------
        param_json : str or dict
            Contains parameters used to generate pattern. 
        
        **kwargs
        num_term : int
            Number of terms in the triangle wave expansion. Must be positive. 
        width, height : angle-like, default unit deg
            Width and height of field of view. Must be positive. 
        spacing : angle-like, default unit deg
            Space between adjacent (parallel) scan lines in the Pong pattern. Must be positive.
        velocity : angle/time-like, default unit deg/s
            Target magnitude of the scan velocity excluding turn-arounds. 
        angle : angle-like, default 0, default unit deg
            Position angle of the box in the native coordinate system. 
        sample_interval : time, default 1/400, default unit s
            Time between read-outs. Must be positive.
        max_scan_duration : time-like, default unit sec
            Maximum total scan time to determine number of repeats. Must be positive. 
            Cannot be used with num_repeat.
        num_repeat : int, default 1 
            Number of repeats of the pattern. Must be >= 1
            Cannot be used with max_scan_duration.
        """

        # pass kwargs
        if param_json is None:
            self.param = self._clean_param(**kwargs)
            self.data = self._generate_scan()

        # pass parameters by json
        else:

            if isinstance(param_json, str):
                with open(param_json, 'r') as f:
                    param = json.load(f)
            elif isinstance(param_json, dict):
                param = param_json
            else:
                raise TypeError('param_json')

            # overwrite any parameters
            if 'max_scan_duration' in kwargs.keys():
                param.pop('num_repeat')

            param.update(kwargs)
            self.param = self._clean_param(**param)

            self.data = self._generate_scan()

    def _clean_param(self, **kwargs):
        kwargs = super()._clean_param(**kwargs)
        kwargs['num_term'] = int(kwargs['num_term'])
        kwargs['width'] = u.Quantity(kwargs['width'], u.deg).value
        kwargs['height'] = u.Quantity(kwargs['height'], u.deg).value
        kwargs['spacing'] = u.Quantity(kwargs['spacing'], u.deg).value
        kwargs['velocity'] = u.Quantity(kwargs['velocity'], u.deg/u.s).value
        kwargs['angle'] = u.Quantity(kwargs.get('angle', 0), u.deg).value
        kwargs['sample_interval'] = u.Quantity(kwargs.get('sample_interval', 1/400), u.s).value
        return kwargs

    def _generate_scan(self):
        
        # unpack parameters
        num_term = self.num_term
        width = self.width.value
        height = self.height.value
        spacing = self.spacing.value
        velocity = self.velocity.value
        sample_interval = self.sample_interval.value

        angle = self.angle.to(u.rad).value

        # --- START OF ALGORITHM ---

        # Determine number of vertices (reflection points) along each side of the
        # box which satisfies the common-factors criterion and the requested size / spacing    

        vert_spacing = sqrt(2)*spacing
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

        vavg = velocity/sqrt(2) # changed so velocity is TOTAL velocity and vavg is single-direction velocity
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

        for i in range(pongcount):
            x_coord1 = self._fourier_expansion(num_term, amp_x, t_count, peri_x)
            y_coord1 = self._fourier_expansion(num_term, amp_y, t_count, peri_y)

            x_coord.append(x_coord1*cos(angle) - y_coord1*sin(angle))
            y_coord.append(x_coord1*sin(angle) + y_coord1*cos(angle))
            time_offset.append(t_count)
            t_count += sample_interval
        
        # repeat pattern if necessary 
        data = pd.DataFrame({
            'time_offset': time_offset, 
            'x_coord': x_coord, 'y_coord': y_coord,
        })

        return self._repeat_scan(data)
    
    def _fourier_expansion(self, num_term, amp, t_count, peri):
        N = num_term*2 - 1
        a = (8*amp)/(pi**2)
        b = 2*pi/peri

        pos = 0
        for n in range(1, N+1, 2):
            c = math.pow(-1, (n-1)/2)/n**2 
            pos += c * sin(b*n*t_count)

        pos *= a
        return pos

class Daisy(SkyPattern):
    """See "CV Daisy - JCMT small area scanning patter" (JCMT TCS/UN/005) for details of implementation."""

    _param_unit = {
        'velocity': u.deg/u.s, 'start_acc': u.deg/u.s/u.s, 
        'R0': u.deg, 'Rt': u.deg, 'Ra': u.deg,
        'T': u.s, 'sample_interval': u.s, 'y_offset': u.deg
    }
    repeatable=False

    def __init__(self, param_json=None, **kwargs) -> None:
        """
        Initialize a daisy pattern by passing a parameter file or dictionary and optionally 
        adding keywords to overwrite existing ones:
            option1 : Daisy(param_json, **kwargs) 
        or building from scratch: 
            option2 : Daisy(**kwargs)

        Parameters
        ---------------------------
        param_json : str or dict
            Contains parameters used to generate pattern. 

        **kwargs
        velocity : angle-like/time-like, default unit deg/s
            Constant velocity (CV) for scan to go at. 
        start_acc : acceleration-like, default unit deg/s^2
            Acceleration at start of pattern.
        R0 : angle-like, default unit deg
            Radius R0. Must be positive.
        Rt : angle-like, default unit deg
            Turn radius. Must be positive.
        Ra : angle-like, default unit deg
            Avoidance radius. Must be non-negative. 
        T : time-like, default unit sec
            Total time of the simulation. Must be postivie. 
        sample_interval : time-like, default 1/400, default unit sec
            Time step. 
        y_offset : angle-like, default 0, default unit deg
            Start offset in y. 
        """

        # pass kwargs
        if param_json is None:
            self.param = self._clean_param(**kwargs)
            self.data = self._generate_scan()

        # pass parameters by json
        else:

            if isinstance(param_json, str):
                with open(param_json, 'r') as f:
                    param = json.load(f)
            elif isinstance(param_json, dict):
                param = param_json
            else:
                raise TypeError('param_json')

            param.update(kwargs)
            self.param = self._clean_param(**param)

            self.data = self._generate_scan()

    def _clean_param(self, **kwargs):
        kwargs['velocity'] = u.Quantity(kwargs['velocity'], u.deg/u.s).value
        kwargs['start_acc'] = u.Quantity(kwargs['start_acc'], u.deg/u.s/u.s).value
        kwargs['R0'] = u.Quantity(kwargs['R0'], u.deg).value
        kwargs['Rt'] = u.Quantity(kwargs['Rt'], u.deg).value
        kwargs['Ra'] = u.Quantity(kwargs['Ra'], u.deg).value
        kwargs['T'] = u.Quantity(kwargs['T'], u.s).value
        kwargs['sample_interval'] = u.Quantity(kwargs.get('sample_interval', 1/400), u.s).value
        kwargs['y_offset'] = u.Quantity(kwargs.get('y_offset', 0), u.deg).value
        return kwargs
        
    def _generate_scan(self):

        # unpack parameters
        speed = self.velocity.to(u.arcsec/u.s).value
        start_acc = self.start_acc.to(u.arcsec/u.s/u.s).value
        R0 = self.R0.to(u.arcsec).value
        Rt = self.Rt.to(u.arcsec).value
        Ra = self.Ra.to(u.arcsec).value
        T = self.T.to(u.s).value
        dt = self.sample_interval.to(u.s).value
        y_offset = self.y_offset.to(u.arcsec).value

        # --- START OF ALGORITHM ---

        # Tangent vector & start value
        (vx, vy) = (1.0, 0.0) 

        # Position vector & start value
        (x, y) = (0.0, y_offset) 

        # number of steps 
        N = int(T/dt)

        # x, y arrays for storage
        x_coord = np.empty(N)
        y_coord = np.empty(N)
        #x_vel = np.empty(N)
        #y_vel = np.empty(N)
        
        # Effective avoidance radius so Ra is not used if Ra > R0 
        #R1 = min(R0, Ra) 

        s0 = speed 
        speed = 0 
        for step in range(N): 

            # Ramp up speed with acceleration start_acc 
            # to limit startup transients. Telescope has zero speed at startup. 
            speed += start_acc*dt 
            if speed >= s0: 
                speed = s0 

            r = sqrt(x*x + y*y) 

            # Straight motion inside R0 
            if r < R0: 
                x += vx*speed*dt 
                y += vy*speed*dt 

            # Motion outside R0
            else: 
                (xn,yn) = (x/r,y/r) # Compute unit radial vector 

                # If aiming close to center, resume straight motion
                # seems to only apply for the initial large turn 
                if (-xn*vx - yn*vy) > sqrt(1 - Ra*Ra/r/r): #if (-xn*vx - yn*vy) > 1/sqrt(1 + (Ra/r)**2):
                    x += vx*speed*dt 
                    y += vy*speed*dt 

                # Otherwise decide turning direction
                else: 
                    if (-xn*vy + yn*vx) > 0: 
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
            x_coord[step] = x
            y_coord[step] = y
            #x_vel[step] = speed*vx
            #y_vel[step] = speed*vy

        """
        ax = -2*xval[1: -1] + xval[0:-2] + xval[2:] # numerical acc in x 
        ay = -2*yval[1: -1] + yval[0:-2] + yval[2:] # numerical acc in y 
        x_acc = np.append(np.array([0]), ax/dt/dt)
        y_acc = np.append(np.array([0]), ay/dt/dt)
        x_acc = np.append(x_acc, 0)
        y_acc = np.append(y_acc, 0)
        """

        return pd.DataFrame({
            'time_offset': np.arange(0, T, dt), 
            'x_coord': x_coord/3600, 'y_coord': y_coord/3600, 
            #'x_vel': x_vel/3600, 'y_vel': y_vel/3600,
        })
 
#######################
#  TELESCOPE PATTERN 
#######################

class TelescopePattern():
    """
    Attributes
    -------------------------------
    sky_pattern
    instrument
    param: ra, dec, location, start_hrang or start_datetime
    module_loc
    """

    _param_unit = {'ra': u.deg, 'dec': u.deg, 'start_hrang': u.hourangle, 'start_datetime': u.dimensionless_unscaled, 'location': u.dimensionless_unscaled}
    _data_unit = {'time_offset': u.s, 'alt_coord': u.deg, 'az_coord': u.deg, 'hour_angle': u.hourangle}

    def __init__(self, sky_pattern, instrument, module=None, obs_param=None, **kwargs) -> None:
        """
        Determine the motion of the telescope.
            option1: give sky_pattern, instrument + module, ra, dec, start_datetime, location
            option2: give sky_pattern, instrument + module, ra, dec, start_hrang, location
            option3: 
                instrument + module 
                sky_pattern 
                obs_param = {ra:, dec:, start_:, location: {lat:, lon:, height:, }}        
        """
        
        # get SkyPattern object
        assert(isinstance(sky_pattern, SkyPattern))
        self.sky_pattern = sky_pattern

        # get Instrument object 
        assert(isinstance(instrument, Instrument))
        self.instrument = instrument

        # pass observation data by dict/json file
        if not obs_param is None:

            if isinstance(obs_param, str):
                with open(obs_param, 'r') as f:
                    param = json.load(f)
            elif isinstance(obs_param, dict):
                param = obs_param

            # overwrite parameters
            param.update(kwargs)
            self.param = self._clean_param(**param)
        
        # pass by kwargs
        else:
            self.param = self._clean_param(**kwargs)

        # get alt/az coordinates
        self.recenter(module)

    def _clean_param(self, **kwargs):
        kwargs['ra'] = u.Quantity(kwargs['ra'], u.deg).value
        kwargs['dec'] = u.Quantity(kwargs['dec'], u.deg).value

        if not isinstance(kwargs['location'], EarthLocation):
            location = kwargs['location']
            lat = u.Quantity(location['lat'], u.deg).value
            lon = u.Quantity(location['lon'], u.deg).value
            height = u.Quantity(location['height'], u.m).value
            kwargs['location'] = EarthLocation(lat=lat, lon=lon, height=height)

        if 'start_hrang' in kwargs.keys():
            assert(not 'start_datetime' in kwargs.keys())
            kwargs['start_hrang'] = u.Quantity(kwargs['start_hrang'], u.hourangle).value
        else:
            kwargs['start_datetime'] = pd.Timestamp(kwargs['start_datetime']).to_pydatetime()

        return kwargs

    def _true_module_loc(self, module):
        """ Gets the (dist, theta) of the module from the boresight (not necessarily central tube)"""

        # passed by module identifier or instrument slot name
        if isinstance(module, str):
            try:
                module = self.instrument.get_location(module)
            except ValueError:
                try:
                    module = self.instrument.slots[module]*u.deg
                except KeyError:
                    raise ValueError(f'{module} is not an existing module name or instrument slot')
        else:
            module = [u.Quantity(module[0], u.deg), u.Quantity(module[1], u.deg)]

        dist = module[0].value
        theta = module[1].to(u.rad).value

        # instrument rotation and offset
        instr_x = self.instrument.instr_offset[0].value
        instr_y = self.instrument.instr_offset[1].value
        instr_rot = self.instrument.instr_rot.to(u.rad).value

        # get true module location in terms of x and y
        mod_x = dist*cos(theta)
        mod_y = dist*sin(theta)

        x_offset = mod_x*cos(instr_rot) - mod_y*sin(instr_rot) + instr_x
        y_offset = mod_x*sin(instr_rot) + mod_y*cos(instr_rot) + instr_y

        new_dist = sqrt(x_offset**2 + y_offset**2)
        new_theta = math.degrees(math.atan2(y_offset, x_offset))

        return new_dist, new_theta

    # TRANSFORMATIONS

    def recenter(self, new_module):
        """
        (Re)calculate azimuth/elevation coordinates centered on new module. 

        Parameters
        -----------------------
        new_module : str or (distance, theta)
            string indicating a module name in the instrument e.g. 'SFH'
            string indicating one of the default slots in the instrument e.g. 'c', 'i1'
            tuple of (distance, theta) indicating module's offset from the center of the instrument, default unit deg
        """

        # --- CONVERT SKY PATTERN INTO AZ/ALT COORDINATES ---

        if 'start_datetime' in self.param.keys():
            df_datetime = pd.to_timedelta(self.sky_pattern.time_offset.value, unit='sec') + self.start_datetime

            # get alt/az
            obs = SkyCoord(ra=self.sky_pattern.x_coord + self.ra, dec=self.sky_pattern.y_coord + self.dec, frame='icrs')
            print('Converting to altitude/azimuth...')
            obs = obs.transform_to(AltAz(obstime=df_datetime, location=self.location))
            print('...Converted!')

            # get hour angle
            obs_time = Time(df_datetime, location=self.location)
            hour_angle = obs_time.sidereal_time('apparent').hourangle*u.hourangle - (self.sky_pattern.x_coord + self.ra)
            hour_angle = hour_angle.to(u.hourangle)

            time_offset = self.sky_pattern.time_offset.value
            az_coord = obs.az.deg
            alt_coord = obs.alt.deg
            hour_angle = hour_angle.value

        else:
            # find beginning sidereal time
            SIDEREAL_TO_UT1 = 1.002737909350795
            start_lst = self.start_hrang + self.sky_pattern.x_coord[0] + self.ra
            lst = self.sky_pattern.time_offset.value/3600*SIDEREAL_TO_UT1*u.hourangle + start_lst

            # find hour angle
            hour_angle = lst - (self.sky_pattern.x_coord + self.ra)
            hour_angle = hour_angle.to(u.hourangle)

            # get alt/az
            dec_rad = (self.sky_pattern.y_coord + self.dec).to(u.rad).value
            hour_angle_rad = hour_angle.to(u.rad).value
            lat_rad = self.location.lat.rad

            alt_rad = np.arcsin( np.sin(dec_rad)*sin(lat_rad) + np.cos(dec_rad)*cos(lat_rad)*np.cos(hour_angle_rad) )
            cos_a = (np.sin(dec_rad) - np.sin(alt_rad)*sin(lat_rad)) / (np.cos(alt_rad)*cos(lat_rad))
            az_rad = np.arccos(cos_a)
            
            mask = np.sin(hour_angle_rad) >= 0 
            az_rad[mask] = 2*pi - az_rad[mask]

            time_offset = self.sky_pattern.time_offset.value
            az_coord = np.degrees(az_rad)
            alt_coord = np.degrees(alt_rad)
            hour_angle = hour_angle.value

        # ---  CONVERT AZ/ALT OF MODULE TO AZ/ALT OF BORESIGHT ---
        # currently, az_rad and alt_rad is the motion the given module
        # we need to store the motion for the boresight

        dist, theta = self._true_module_loc(new_module)
        self.module_loc = (dist, theta)

        az_coord, alt_coord = self._transform_to_boresight(az_coord, alt_coord, dist, theta)

        # save important data
        self.data = pd.DataFrame({
            'time_offset': time_offset,
            'az_coord': az_coord, 'alt_coord': alt_coord, 'hour_angle': hour_angle
        })

    def _transform_to_boresight(self, az1, alt1, dist, theta):

        # convert everything into radians
        dist = math.radians(dist)
        theta = math.radians(theta)
        alt1 = np.radians(alt1)
        az1 = np.radians(az1)

        # getting new elevation
        def func(alt_0, alt_1):
            return sin(alt_0)*cos(dist) + sin(dist)*cos(alt_0)*sin(theta + alt_0) - sin(alt_1)

        alt0 = np.empty(len(alt1))
        guess = alt1[0]
        for i, a1 in enumerate(alt1):
            try:
                a0 = root_scalar(func, args=(a1), x0=guess, bracket=[-pi/2, pi/2], xtol=10**(-6)).root
                guess = a0
            except ValueError:
                print(f'nan value at {i}')
                alt0 = math.nan

            alt0[i]= a0

        # getting new azimuth
        #cos_diff_az0 = ( np.cos(alt0)*cos(dist) - np.sin(alt0)*sin(dist)*np.sin(theta + alt0) )/np.cos(alt1)
        cos_diff_az0 = (cos(dist) - np.sin(alt0)*np.sin(alt1))/(np.cos(alt0)*np.cos(alt1))
        cos_diff_az0[cos_diff_az0 > 1] = 1
        diff_az0 = np.arccos(cos_diff_az0)

        # check if diff_az is positive or negative
        mask = (theta > -alt0 + pi/2) & (theta < -alt0 + 3*pi/2)
        diff_az0[mask] = -diff_az0[mask]
        az0 = az1 - diff_az0

        """# check is dist is dist
        beta = np.arcsin( np.cos(theta + alt0)*np.cos(alt0)/np.cos(alt1) )
        alpha = pi/2 - beta + theta + alt0
        dist_check = np.degrees(np.arcsin( np.cos(alt1)*np.sin(alpha)/np.cos(theta + alt0)  ))
        plt.plot(dist_check)"""

        return np.degrees(az0)%360, np.degrees(alt0)%360

    def _transform_from_boresight(self, az0, alt0, dist, theta):
        
        # convert everything into radians
        dist = math.radians(dist)
        theta = math.radians(theta)
        alt0 = np.radians(alt0)
        az0 = np.radians(az0)

        # find elevation offset
        alt1 = np.arcsin(np.sin(alt0)*cos(dist) + sin(dist)*np.cos(alt0)*np.sin(theta + alt0))

        # find azimuth offset
        sin_az1 = 1/np.cos(alt1) * ( np.cos(alt0)*np.sin(az0)*cos(dist) + np.cos(az0)*np.cos(theta + alt0)*sin(dist) - np.sin(alt0)*np.sin(az0)*sin(dist)*np.sin(theta + alt0) )
        cos_az1 = 1/np.cos(alt1) * ( np.cos(alt0)*np.cos(az0)*cos(dist) - np.sin(az0)*np.cos(theta + alt0)*sin(dist) - np.sin(alt0)*np.cos(az0)*sin(dist)*np.sin(theta + alt0) )
        az1 = np.arctan2(sin_az1, cos_az1)

        return np.degrees(az1)%360, np.degrees(alt1)%360

    def view_module(self, module):
        """
        Parameters
        ------------------------
        module : str or (distance, theta)
            string indicating a module name in the instrument e.g. 'SFH'
            string indicating one of the default slots in the instrument e.g. 'c', 'i1'
            tuple of (distance, theta) indicating module's offset from the center of the instrument
        """

        dist, theta = self._true_module_loc(module)
        az1, alt1 = self._transform_from_boresight(self.az_coord.value, self.alt_coord.value, dist, theta)
        return az1, alt1

    # SAVING/EXTRACTING DATA

    def save_param(self, param_json=None):
        """
        Parameters
        ----------------------------
        param_json : str or False
            path to intended file location for the parametes
            if None, return it as a dictionary
        """

        param_temp = self.param.copy()
        location = param_temp['location']
        param_temp['location'] = {'lat': location.lat.deg, 'lon': location.lon.deg, 'height': location.height.value}

        # save param_json
        if param_json is None:
            return param_temp
        else:
            with open(param_json, 'w') as f:
                json.dump(param_temp, f)

    def save_data(self, path_or_buf=None, columns='default'):
        """
        Parameters
        ----------------------------
        path_or_buf : str or file handle, default None
            File path or object, if None is provided the result is returned as a dictionary.
        columns : sequence or str, default 'default'
            Columns to write. 
            'default' for ['time_offset', 'az_coord', 'alt_coord', 'az_vel', 'alt_vel']
            'all' for ['time_offset', 'az_coord', 'alt_coord', 'az_vel', 'alt_vel', 'vel', 'az_acc', 'alt_acc', 'acc', 'az_jerk', 'alt_jerk', 'jerk', 'hour_angle', 'para_angle', 'rot_angle']
        """

        # replace str options
        if columns == 'default':
            columns = ['time_offset', 'az_coord', 'alt_coord', 'az_vel', 'alt_vel']
        elif columns == 'all':
            columns = ['time_offset', 'az_coord', 'alt_coord', 'az_vel', 'alt_vel', 'vel', 'az_acc', 'alt_acc', 'acc', 'az_jerk', 'alt_jerk', 'jerk', 'hour_angle', 'para_angle', 'rot_angle']
        
        # generate required data
        data = self.data.copy()
        for col in columns:
            if not col in ['time_offset', 'az_coord', 'alt_coord']:
                data[col] = getattr(self, col)
        
        # save data file 
        if path_or_buf is None:
            return data[columns].to_dict()
        else:
            data.to_csv(path_or_buf, columns=columns, index=False)

    # ATTRIBUTES

    def __getattr__(self, attr):

        if attr in self.param.keys():
            if self._param_unit[attr] is u.dimensionless_unscaled:
                return self.param[attr]
            else:
                return self.param[attr]*self._param_unit[attr]
        elif attr in self.data.columns:
            return self.data[attr].to_numpy()*self._data_unit[attr]
        else:
            raise AttributeError(f'attribtue {attr} not found')

    @property
    def az_vel(self):
        return _central_diff(self.az_coord.value, self.sky_pattern.sample_interval.value)*u.deg/u.s

    @property
    def alt_vel(self):
        return _central_diff(self.alt_coord.value, self.sky_pattern.sample_interval.value)*u.deg/u.s

    @property
    def vel(self):
        return np.sqrt(self.az_vel**2 + self.alt_vel**2)

    @property
    def az_acc(self):
        return _central_diff(self.az_vel.value, self.sky_pattern.sample_interval.value)*u.deg/u.s/u.s

    @property
    def alt_acc(self):
        return _central_diff(self.alt_vel.value, self.sky_pattern.sample_interval.value)*u.deg/u.s/u.s
    
    @property
    def acc(self):
        return np.sqrt(self.az_acc**2 + self.alt_acc**2)

    @property
    def az_jerk(self):
        return _central_diff(self.az_acc.value, self.sky_pattern.sample_interval.value)*u.deg/(u.s)**3
    
    @property
    def alt_jerk(self):
        return _central_diff(self.alt_acc.value, self.sky_pattern.sample_interval.value)*u.deg/(u.s)**3
    
    @property
    def jerk(self):
        return np.sqrt(self.az_jerk**2 + self.alt_jerk**2)

    @property
    def para_angle(self):
        dec_rad = self.dec.to(u.rad).value
        hour_angle_rad = self.hour_angle.to(u.rad).value
        lat_rad = self.location.rad
        
        return np.degrees(np.arctan2( 
            np.sin(hour_angle_rad), 
            cos(dec_rad)*tan(lat_rad) - sin(dec_rad)*np.cos(hour_angle_rad) 
        ))*u.deg

    @property
    def rot_angle(self):
        return self.para_angle + self.alt_coord

# visibility function, getting hour angles when object has appropriate airmass, elevation/scale_factor, field rotation angle rate
# plotting functions