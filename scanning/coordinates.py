import math
from math import pi, sin, cos, tan, sqrt, radians
import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation

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

FYST_LOC = EarthLocation(lat='-22d59m08.30s', lon='-67d44m25.00s', height=5611.8*u.m)

##################
#  SKY PATTERN 
##################

class SkyPattern():
    """ Representing the path of the center of a detector array in terms of offsets. """

    # other attributes (for development)
    # _stored_units
    # _data: time_offset, x_coord, y_coord
    # _param, _param_units
    # _sample_interval
    # _repeatable

    _param_units = {'num_repeat': u.dimensionless_unscaled}
    _stored_units = {'time_offset': u.s, 'x_coord': u.deg, 'y_coord': u.deg}

    def __init__(self, data, units=None, repeatable=False, **kwargs) -> None:
        """
        Parameters
        --------------------------------
        data : str, DataFrame, or [dict of str -> sequence]
            If `str`, a file path to a csv file. If `dict` or `DataFrame`, column names map to their values. 
            Must have columns 'time_offset', 'x_coord', 'y_coord'.
        repeatable : bool; default False
            Whether this pattern can repeat itself (ends where it starts).
        units : [dict of str -> str or Unit] or None; default None
            Mapping columns in `data` with their units. All columns do not need to be mapped.
            If not provided, all angle-like units are assumed to be in degrees and all time-like units are assumed to be in seconds.
        
        Keyword Args
        ---------------------------------
        num_repeat : int; default 1 
            Number of repeats of the pattern. Must be a positive integer.
            Cannot be used with `max_scan_duration` and `repeatable` must be `True`.
        max_scan_duration : float/Quantity/str; default unit sec
            Maximum total scan time to determine number of repeats. Must be positive. 
            Cannot be used with `num_repeat` and `repeatable` must be `True`.

        Raises
        -----------------------------------
        ValueError
            "data" could not be parsed
        ValueError
            sample interval must be constant

        Examples
        -----------------------------
        >>> # if x_coord is specifically in arcseconds
        >>> SkyPattern('file.csv', units={'x_coord': 'arcsec'})
        """

        self._repeatable = repeatable 

        try:
            if isinstance(data, str):
                data = pd.read_csv(data, index_col=False, usecols=['time_offset', 'x_coord', 'y_coord'])
            else:
                data = pd.DataFrame(data)[['time_offset', 'x_coord', 'y_coord']]
        except (ValueError, KeyError) as e:
            raise ValueError(f"'data' could not be parsed: {e}")

        # convert to specified units
        if not units is None:
            for col, unit in units.items():
                data[col] = data[col]*u.Unit(unit).to(self._stored_units[col])

        # determine sample_interval 
        sample_interval_list = np.diff(data['time_offset'].to_numpy())
        if np.std(sample_interval_list)/np.mean(sample_interval_list) <= 0.01:
            self._sample_interval = np.mean(sample_interval_list)
        else:
            raise ValueError('sample interval must be constant')

        # repeating the scan
        self._param = self._clean_param(**kwargs)
        self._data = self._repeat_scan(data)

    # INITIALIZATION

    def _clean_param(self, **kwargs):
        kwarg_keys = kwargs.keys()

        # determine number of repeats
        if 'max_scan_duration' in kwarg_keys or 'num_repeat' in kwarg_keys:

            if not self.repeatable:
                warnings.warn('This is not a repeatable pattern, but you have indicated to repeat it. This may or may not repeat.')
            
            if 'max_scan_duration' in kwarg_keys and 'num_repeat' in kwarg_keys:
                raise ValueError('"max_scan_duration" and "num_repeat" cannot be inputted together')

            if 'max_scan_duration' in kwarg_keys:
                kwargs['num_repeat'] = math.nan # set as null for now, will determine once first pattern is generated
                kwargs['max_scan_duration'] = u.Quantity(kwargs['max_scan_duration'], self._stored_units['time_offset']).value
            else:
                kwargs['num_repeat'] = int(kwargs.get('num_repeat', 1))

        return kwargs

    def _repeat_scan(self, data):
        one_scan_duration = data.iloc[-1]['time_offset'] + self.sample_interval.value

        num_repeat = self._param.get('num_repeat', 1)

        # determine number of repeats
        if math.isnan(num_repeat):
            max_scan_duration = self._param.pop('max_scan_duration') # only store number of repeats, not maximum scan duration
            num_repeat = math.floor(max_scan_duration/one_scan_duration)
            self._param['num_repeat'] = num_repeat

        # repeat pattern if necessary 
        if num_repeat < 1:
            raise ValueError(f'number of repeats = {num_repeat} is less than 1')
        elif num_repeat > 1:
            warnings.warn('You have chosen to repeat this pattern. However, analytic patterns \
            (such as Pong) may not have their next location in the repeat be exactly their first original location \
            to ensure a constant time interval. This may cause spikes in higher derivates. Mitigate this by \
            initializing this object with its intended subclass using its original parameters.')

            time_offset = data['time_offset']
            x_coord = data['x_coord']
            y_coord = data['y_coord']
            data_temp = data.copy()
            for i in range(1, num_repeat):
                data_temp['time_offset'] = time_offset + one_scan_duration*i
                data_temp['x_coord'] = x_coord
                data_temp['y_coord'] = y_coord
                data = data.append(data_temp, ignore_index=True)

        return data

    # OBJECT DATA

    def save_data(self, path_or_buf=None, columns='default', include_repeats=True):
        """

        Parameters
        ----------------------
        path_or_buf : str, file handle or None; default None
            File path or object, if `None` is provided the result is returned as a dictionary.
        columns : sequence, str or [dict of str -> str/Unit/None]; default 'default'
            Columns to write. If `dict`, map column names to their desired unit and use `None` if you would like to use the standard (deg for angle-like units, sec for time-like units).
            'default' for ['time_offset', 'x_coord', 'y_coord']
            'all' for ['time_offset', 'x_coord', 'y_coord', 'distance', 'x_vel', 'y_vel', 'vel', 'x_acc', 'y_acc', 'acc', 'x_jerk', 'y_jerk', 'jerk']
        include_repeats : bool, default 'True'
            Whether to include repeats of the SkyPattern.

        Returns
        ----------------------
        None or [dict of str -> array]
            If `path_or_buf` is `None`, returns the data as a dictionary mapping column name to values. Otherwise returns `None`.

        Examples
        ---------------------
        >>> skypattern.save_data('file.csv', columns={'time_offset': 's', 'x_coord': 'arcsec', 'y_coord': None})
        """

        # replace str options
        if columns == 'default':
            columns = ['time_offset', 'x_coord', 'y_coord']
        elif columns == 'all':
            columns = ['time_offset', 'x_coord', 'y_coord', 'distance', 'x_vel', 'y_vel', 'vel', 'x_acc', 'y_acc', 'acc', 'x_jerk', 'y_jerk', 'jerk']
        
        data = pd.DataFrame()

        # generate required data
        if isinstance(columns, dict):
            for col, unit in columns.items():
                if not unit is None:
                    data[col] = getattr(self, col).to(unit).value
                else:
                    data[col] = getattr(self, col).value
        else:
            for col in columns:
                data[col] = getattr(self, col).value

        # whether to include repetitions 
        num_repeat = self._param.get('num_repeat', 1)
        if not include_repeats and num_repeat > 1:
            before_index = int(len(data.index)/num_repeat)
            data = data.iloc[:before_index]
        
        # returning
        if path_or_buf is None:
            return data.to_dict('list')
        else:
            data.to_csv(path_or_buf, index=False)

    def save_param(self, path_or_buf=None):
        """
        Parameters
        ----------------------------
        path_or_buf : str, file handle, or None; default None
            File path or object, if `None` is provided the result is returned as a dictionary.
        
        Returns
        ----------------------
        None or [dict of str -> numeric]
            If `path_or_buf` is `None`, returns the resulting json format as a dictionary. Otherwise returns `None`.
        """
        
        param_temp = self._param.copy()

        # save param_json
        if path_or_buf is None:
            return param_temp
        else:
            with open(path_or_buf, 'w') as f:
                json.dump(param_temp, f)

    # PROPERTIES
    
    def __getattr__(self, attr):
        # for easy access of properties without unit conversions
        if attr.startswith('_'):
            prop = getattr(self, attr[1:])
            if type(prop) is u.Quantity:
                return prop.value
            else:
                return prop
        else:
            raise AttributeError(f'type object "{type(self)}" has no attribute "{attr}"')

    @property
    def repeatable(self):
        """bool: Whether this pattern is repeatable or not."""
        return self._repeatable

    @property
    def param(self):
        """dict of str -> (float or Quantity): Parameters inputted by user."""
        return_param = dict()
        for p, val in self._param.items():
            return_param[p] = val if self._param_units[p] is u.dimensionless_unscaled else val*self._param_units[p]
        return return_param

    @property
    def sample_interval(self):
        """Quantity: Time interval between samples."""
        return self._sample_interval*self._stored_units['time_offset']

    @property
    def scan_duration(self):
        """Quantity: Total scan duration."""
        return self.time_offset[-1] + self.sample_interval

    @property
    def time_offset(self):
        """Quantity array: Time offsets."""
        return self._data['time_offset'].to_numpy()*self._stored_units['time_offset']
    
    @property
    def x_coord(self):
        """Quantity array: x positions."""
        return self._data['x_coord'].to_numpy()*self._stored_units['x_coord']
    
    @property
    def y_coord(self):
        """Quantity array: y positions."""
        return self._data['y_coord'].to_numpy()*self._stored_units['y_coord']

    @property
    def distance(self):
        """Quantity array: Distance of points from the center."""
        return np.sqrt(self.x_coord**2 + self.y_coord**2)

    @property
    def x_vel(self):
        """Quantity array: x velocity."""
        return _central_diff(self.x_coord.value, self.sample_interval.value)*(self._stored_units['x_coord']/self._stored_units['time_offset'])

    @property
    def y_vel(self):
        """Quantity array: y velocity."""
        return _central_diff(self.y_coord.value, self.sample_interval.value)*(self._stored_units['y_coord']/self._stored_units['time_offset'])

    @property
    def vel(self):
        """Quantity array: Total velocity."""
        return np.sqrt(self.x_vel**2 + self.y_vel**2)

    @property
    def x_acc(self):
        """Quantity array: x acceleration."""
        return _central_diff(self.x_vel.value, self.sample_interval.value)*(self._stored_units['x_coord']/self._stored_units['time_offset']**2)

    @property
    def y_acc(self):
        """Quantity array: y acceleration."""
        return _central_diff(self.y_vel, self.sample_interval.value)*(self._stored_units['y_coord']/self._stored_units['time_offset']**2)
    
    @property
    def acc(self):
        """Quantity array: Total acceleration."""
        return np.sqrt(self.x_acc**2 + self.y_acc**2)

    @property
    def x_jerk(self):
        """Quantity array: x jerk."""
        return _central_diff(self.x_acc.value, self.sample_interval.value)*(self._stored_units['x_coord']/self._stored_units['time_offset']**3)
    
    @property
    def y_jerk(self):
        """Quantity array: y jerk."""
        return _central_diff(self.y_acc.value, self.sample_interval.value)*(self._stored_units['y_coord']/self._stored_units['time_offset']**3)
    
    @property
    def jerk(self):
        """Quantity array: Total jerk."""
        return np.sqrt(self.x_jerk**2 + self.y_jerk**2)

class Pong(SkyPattern):
    """
    The Curvy Pong pattern allows for an approximation of a Pong pattern while avoiding 
    sharp turnarounds at the vertices. The Pong pattern is an analytic and close-pathed 
    pattern that is optimized for regions a few square degrees. It makes a path that 
    intends to cover each area uniformly and ends where it starts.
    
    See "The Impact of Scanning Pattern Strategies on Uniform Sky Coverage of Large Maps" 
    (SCUBA Project SC2/ANA/S210/008) for details of implementation. 
    """

    _repeatable = True
    _param_units = {
        'num_term': u.dimensionless_unscaled,
        'width': u.deg, 'height': u.deg, 'spacing': u.deg,
        'velocity': u.deg/u.s, 'angle': u.deg, 'sample_interval': SkyPattern._stored_units['time_offset'], 
        'num_repeat': u.dimensionless_unscaled
    }

    def __init__(self, param_json=None, **kwargs) -> None:
        """
        Initialize a Pong pattern by passing a parameter file and overwriting any parameters with **kwargs:
            option1 : Pong(param_json, **kwargs) 
        or building from scratch: 
            option2 : Pong(**kwargs)

        Parameters
        ---------------------------
        param_json : str or None
            If `str` path to JSON file containing parameters. 
        
        Keyword Args
        ----------------------------
        num_term : int
            Number of terms in the triangle wave expansion. Must be positive. 
        width : float or Quantity or str; default unit deg
            Width of the field. Must be positive. 
        height : float or Quantity or str; default unit deg
            Height of the field. Must be positive. 
        spacing : float or Quantity or str; default unit deg
            Space between adjacent (parallel) scan lines in the Pong pattern. Must be positive.
        velocity : float or Quantity or str; default unit deg/s
            Target magnitude of the total scan velocity excluding turn-arounds.
            NOTE this is now the total velocity, not just the velocity of one direction.

        angle : float or Quantity or str; default 0; default unit deg
            Position angle of the box in the native coordinate system. 
        sample_interval : float or Quantity or str; default 1/400, default unit s
            Time between read-outs. Must be positive.
        num_repeat : int; default 1 
            Number of repeats of the pattern. Must be a positive integer.
            Cannot be used with `max_scan_duration`.
        max_scan_duration : float or Quantity or str; default unit sec
            Maximum total scan time to determine number of repeats. Must be positive. 
            Cannot be used with `num_repeat`.
        
        Examples
        ---------------------------
        >>> import astropy.units as u
        >>> Pong(num_term=4, width=2, height=7200*u.arcsec, spacing='500 arcsec', velocity=1/2)
        """

        # pass kwargs
        if param_json is None:
            self._param = self._clean_param(**kwargs)

        # pass parameters by json
        else:
            with open(param_json, 'r') as f:
                param = json.load(f)
           
            # overwrite any parameters
            if 'max_scan_duration' in kwargs.keys():
                param.pop('num_repeat')
            param.update(kwargs)

            self._param = self._clean_param(**param)
        
        self._sample_interval = self._param['sample_interval']
        self._data = self._generate_scan()

    def _clean_param(self, **kwargs):
        kwargs = super()._clean_param(**kwargs)
        kwargs['num_term'] = int(kwargs['num_term'])
        kwargs['width'] = u.Quantity(kwargs['width'], self._param_units['width']).value
        kwargs['height'] = u.Quantity(kwargs['height'], self._param_units['height']).value
        kwargs['spacing'] = u.Quantity(kwargs['spacing'], self._param_units['spacing']).value
        kwargs['velocity'] = u.Quantity(kwargs['velocity'], self._param_units['velocity']).value
        kwargs['angle'] = u.Quantity(kwargs.get('angle', 0), self._param_units['angle']).value
        kwargs['sample_interval'] = u.Quantity(kwargs.get('sample_interval', 1/400), self._param_units['sample_interval']).value
        return kwargs

    def _generate_scan(self):
        
        # unpack parameters
        num_term = self._param['num_term']
        width = self._param['width']
        height = self._param['height']
        spacing = self._param['spacing']
        velocity = self._param['velocity']
        sample_interval = self._param['sample_interval']

        angle_rad = radians(self._param['angle'])

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

        amp_x = x_numvert * vert_spacing / 2
        amp_y = y_numvert * vert_spacing / 2

        # Determine number of repeats
        num_repeat = self._param.get('num_repeat', 1)

        if math.isnan(num_repeat):
            max_scan_duration = self._param.pop('max_scan_duration') # only store number of repeats, not maximum scan duration
            num_repeat = math.floor(max_scan_duration/period)
            self._param['num_repeat'] = num_repeat

        pongcount = math.ceil(period*num_repeat/sample_interval)
        
        # Calculate the grid positions and apply rotation angle. Load
        # data into a dataframe.    

        t_count = 0
        time_offset = []
        x_coord = []
        y_coord = []

        for i in range(pongcount):
            x_coord1 = self._fourier_expansion(num_term, amp_x, t_count, peri_x)
            y_coord1 = self._fourier_expansion(num_term, amp_y, t_count, peri_y)

            x_coord.append(x_coord1*cos(angle_rad) - y_coord1*sin(angle_rad))
            y_coord.append(x_coord1*sin(angle_rad) + y_coord1*cos(angle_rad))
            time_offset.append(t_count)
            t_count += sample_interval
        
        # repeat pattern if necessary 
        return pd.DataFrame({
            'time_offset': time_offset, 
            'x_coord': x_coord, 'y_coord': y_coord,
        })
    
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
    """
    The Daisy pattern is optimized for point sources and works by having the path of the camera module 
    move at constant velocity and cross the center of the map at various angles.

    See "CV Daisy - JCMT small area scanning pattern" (JCMT TCS/UN/005) for details of implementation.
    """

    _param_units = {
        'velocity': u.deg/u.s, 'start_acc': u.deg/u.s/u.s, 
        'R0': u.deg, 'Rt': u.deg, 'Ra': u.deg,
        'T': u.s, 'sample_interval': SkyPattern._stored_units['time_offset'], 'y_offset': u.deg
    }
    _repeatable = False

    def __init__(self, param_json=None, **kwargs) -> None:
        """
        Initialize a Daisy pattern by passing a parameter file and overwriting any parameters with **kwargs:
            option1 : Daisy(param_json, **kwargs) 
        or building from scratch: 
            option2 : Daisy(**kwargs)

        Parameters
        ---------------------------
        param_json : str or None
            If `str` path to JSON file containing parameters. 
        
        Keyword Args
        ----------------------------
        velocity : float or Quanity or str; default unit deg/s
            Constant velocity (CV) for scan to go at. 
        start_acc : float or Quanity or str; default unit deg/s^2
            Acceleration at start of pattern. Cannot be 0. 
        R0 : float or Quanity or str; default unit deg
            Radius R0. Must be positive.
        Rt : float or Quanity or str; default unit deg
            Turn radius. Must be positive.
        Ra : float or Quanity or str; default unit deg
            Avoidance radius. Must be non-negative. 
        T : float or Quanity or str; default unit sec
            Total time of the simulation. Must be postivie. 
        sample_interval : float or Quanity or str; default 1/400, default unit sec
            Time step. 
        y_offset : float or Quanity or str; default 0, default unit deg
            Start offset in y. 

        Examples
        ----------------------------
        >>> import astropy.units as u
        >>> Daisy(velocity=1/3*u.deg/u.s, start_acc='0.2 deg/s/s', R0=0.47, Rt=800*u.arcsec, Ra='600 arcsec', T=300)
        """

        # pass kwargs
        if param_json is None:
            self._param = self._clean_param(**kwargs)

        # pass parameters by json
        else:
            with open(param_json, 'r') as f:
                param = json.load(f)
           
            # overwrite any parameters
            param.update(kwargs)
            self._param = self._clean_param(**param)
        
        self._sample_interval = self._param['sample_interval']
        self._data = self._generate_scan()

    def _clean_param(self, **kwargs):
        kwargs['velocity'] = u.Quantity(kwargs['velocity'], self._param_units['velocity']).value
        kwargs['start_acc'] = u.Quantity(kwargs['start_acc'], self._param_units['start_acc']).value
        kwargs['R0'] = u.Quantity(kwargs['R0'], self._param_units['R0']).value
        kwargs['Rt'] = u.Quantity(kwargs['Rt'], self._param_units['Rt']).value
        kwargs['Ra'] = u.Quantity(kwargs['Ra'], self._param_units['Ra']).value
        kwargs['T'] = u.Quantity(kwargs['T'], self._param_units['T']).value
        kwargs['sample_interval'] = u.Quantity(kwargs.get('sample_interval', 1/400), self._param_units['sample_interval']).value
        kwargs['y_offset'] = u.Quantity(kwargs.get('y_offset', 0), self._param_units['y_offset']).value
        return kwargs
        
    def _generate_scan(self):

        # unpack parameters
        param = self.param

        speed = param['velocity'].to(u.arcsec/u.s).value
        start_acc = param['start_acc'].to(u.arcsec/u.s/u.s).value
        R0 = param['R0'].to(u.arcsec).value
        Rt = param['Rt'].to(u.arcsec).value
        Ra = param['Ra'].to(u.arcsec).value
        T = param['T'].to(u.s).value
        dt = param['sample_interval'].to(u.s).value
        y_offset = param['y_offset'].to(u.arcsec).value

        # If the sample rate is too low, sample at a higher frequency 
        # and then only take a subset. 

        good_dt = 1/150
        if (dt > good_dt):
            sample_every = math.ceil(dt/good_dt)
            dt = dt/sample_every
        else:
            sample_every = 1

            
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
        test = []
        
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

                    # NOTE converting back into a unit vector, for long interations, the Daisy pattern starts spiraling out otherwise
                    total_v = sqrt(vx**2 + vy**2)
                    vx = vx/total_v
                    vy = vy/total_v

                    test.append(sqrt(vx**2 + vy**2))

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
        
        # check if pattern has spiraled
        total_R = sqrt(R0**2 - 2*Rt*Ra + Rt**2) + Rt
        last_R = sqrt(x_coord[-1]**2 + y_coord[-1]**2)

        if last_R >= total_R + 2*Rt:
            warnings.warn('This Daisy scan may have spiraled out.')

        # return data
        data =  pd.DataFrame({
            'time_offset': np.arange(0, T, dt), 
            'x_coord': x_coord/3600, 'y_coord': y_coord/3600, 
        })

        if sample_every == 1:
            return data
        else:
            return data.iloc[::sample_every, :]
 
#######################
#  TELESCOPE PATTERN 
#######################

class TelescopePattern():
    """
    Representing the path in AZ/EL coordinates of the telescope's boresight.
    """

    # other attributes (for development)
    # _stored_units
    # _instrument
    # _data
    # _param, _param_units

    _param_units = {
        'start_ra': u.deg, 'start_dec': u.deg, 'lat': u.deg,
        'start_hrang': u.hourangle, 
        'start_datetime': u.dimensionless_unscaled, 
        'start_lst': u.hourangle,
        'start_elev': u.deg, 'moving_up': u.dimensionless_unscaled
    }
    _stored_units = {'time_offset': u.s, 'lst': u.hourangle, 'alt_coord': u.deg, 'az_coord': u.deg}

    # INITIALIZATION

    def __init__(self, data, instrument=None, data_loc='boresight', obs_param=None, units=None, **kwargs) -> None:
        """
        Determine the motion of the telescope. Note that **kwargs can be passed as a file through `obs_param`.
            | option1: data (sky), instrument (optional), data_loc (optional); lat, start_ra, start_dec, (start_datetime or start_hrang or start_lst or [start_elev and moving_up])
            | option2: data (telescope, excludes lst), instrument (optional), data_loc (optional); lat, (start_ra or start_datetime or start_lst)
            | option3: data (telescope, includes lst), instrument (optional), data_loc (optional); lat

        Parameters
        -----------------------------
        data : str, DataFrame, [dict of str -> sequence], or SkyPattern
            If `str`, a file path to a csv file. If `dict` or `DataFrame`, column names map to their values. 
            Columns must contain 'time_offset, 'az_coord', and 'alt_coord'. If 'lst' is included as a column, certain observation parameters are not required (see options).
            Otherwise, `data` is a `SkyPattern` object. 
        units : [dict of str -> str or Unit] or None; default None
            If `data` is not `SkyPattern`, this mapping column in `data` with their units. All columns do not need to be mapped.
            If not provided, all angle-like units are assumed to be in degrees (except for hour angle and lst which are in hourangle)
            and all time-like units are assumed to be in seconds.

        instrument : Instrument or None, default None
            An `Instrument` object. 
        data_loc : str or two-tuple; default 'boresight'
            Location relative to the center of the instrument where observation parameters are applied:
                | 1. 'boresight' for boresight of the telescope 
                | 2. string indicating a module name in the instrument e.g. 'SFH' or one of the default slots in the instrument e.g. 'c', 'i1'
                | 3. tuple of (distance, theta) indicating module's offset from the center of the instrument; default unit deg

        obs_param : str
            File path to json containing all required obervation parameters (see **kwargs).

        Keyword Args
        ------------------------------
        start_ra : float/Quantity/str; default unit deg
            Starting right acension of telescope / right acension offset for sky.
        start_dec : float/Quantity/str; default unit deg
            Declination offset for sky_pattern.
        lat : float/Quantity/str; default unit deg, default FYST_LOC.lat
            Latitude of observation. 

        start_datetime : str or datetime; default timezone UTC
            Starting date and time of observation.
        start_hrang : float/Quantity/str; default unit hourangle
            Starting hour angle of source.
        start_lst : float/Quantity/str; default unit hourangle
            Starting local sidereal time.
        start_elev : float/Quantity/str; default unit deg
            Starting elevation of source, must be used with `moving_up`.
        moving_up : bool, default True
            Whether observation is moving towards the meridian or away, must be used with `start_elev`. 

        Example
        -------------------------------
        >>> sky_pattern = SkyPattern(sky_pattern.csv)
        >>> telescope_pattern1 = TelescopePattern(sky_pattern, data_loc='i1', start_ra='5 hourangle', start_dec='-60 deg', start_elev=30)

        """

        # --- Observation Parameters ---

        # pass by obs_param
        if not obs_param is None:
            with open(obs_param, 'r') as f:
                param = json.load(f)

            # overwrite parameters FIXME start_
            param.update(kwargs)

        else:
            param = kwargs

        # --- sky_pattern or data ---

        # sky_pattern has been passed
        if isinstance(data, SkyPattern):
                
            self._param = self._clean_param_sky_pattern(**param)

            self._sample_interval = data.sample_interval.value
            self._data = pd.DataFrame({'time_offset': data.time_offset.value})
            self._data['lst'] = self._get_lst(data)
            az1, alt1 = self._from_sky_pattern(data)

            self._data['az_coord'] = az1
            self._data['alt_coord'] = alt1

        # data has been passed
        else:
            if isinstance(data, str):
                try: 
                    self._data = pd.read_csv(data, index_col=False)
                except ValueError:
                    raise ValueError('could not parse "data"')
            else:
                try:
                    self._data = pd.DataFrame(data)
                except ValueError:
                    raise ValueError('could not parse "data"')

            if not units is None:
                for col, unit in units.items():
                    self._data[col] = self._data[col]*u.Unit(unit).to(self._stored_units[col])
                
            if 'lst' in self._data.columns:
                self._param = self._clean_param_telescope_data(False, **param)
            else:
                self._param = self._clean_param_telescope_data(True, **param)

            try:
                self._data[['time_offset', 'lst', 'az_coord', 'alt_coord']]
            except KeyError:
                self._data[['time_offset', 'az_coord', 'alt_coord']]
                self._data['lst'] = self._get_lst()

            # determine sample_interval 
            sample_interval_list = np.diff(self.time_offset.value)
            if np.std(sample_interval_list)/np.mean(sample_interval_list) <= 0.01:
                sample_interval = np.mean(sample_interval_list)
                self._sample_interval = sample_interval
            else:
                raise ValueError('sample_interval must be constant')

        # --- Instrument and module --- 

        self._instrument = instrument
        dist, theta = self._true_module_loc(data_loc)
        
        # az_coord and alt_coord are for the module's location
        # we need the az/alt coordinates of the boresight

        if not (math.isclose(dist, 0) and math.isclose(theta, 0)):
            az0, alt0 = self._transform_to_boresight(self.az_coord.value, self.alt_coord.value, dist, theta)
            self._data['az_coord'] = az0
            self._data['alt_coord'] = alt0
        else:
            self._data['az_coord'] = self._norm_angle(self.az_coord.value)
        
        if len(self.alt_coord.value[(self.alt_coord.value < 0) | (self.alt_coord.value > 90)]) > 0:
            warnings.warn('elevation has values outside of 0 to 90 range')
        elif len(self.alt_coord.value[(self.alt_coord.value < 30) | (self.alt_coord.value > 75)]) > 0:
            warnings.warn('elevation has values outside of 30 to 75 range')

    def _clean_param_sky_pattern(self, **kwargs):
        kwarg_keys = kwargs.keys()
        new_kwargs = dict()

        # required
        new_kwargs['lat'] = u.Quantity(kwargs.pop('lat', FYST_LOC.lat), u.deg).value
        new_kwargs['start_ra'] = u.Quantity(kwargs.pop('start_ra'), u.deg).value
        new_kwargs['start_dec'] = u.Quantity(kwargs.pop('start_dec'), u.deg).value

        # choose between
        if np.count_nonzero( ['start_hrang' in kwarg_keys, 'start_datetime' in kwarg_keys, 'start_lst' in kwarg_keys, 'start_elev' in kwarg_keys] ) != 1:
            raise TypeError('need one (and only one) of start_datetime, start_hrang, start_lst, or start_elev')

        if 'start_hrang' in kwarg_keys:
            new_kwargs['start_hrang'] = u.Quantity(kwargs.pop('start_hrang'), u.hourangle).value
        elif 'start_datetime' in kwarg_keys:
            new_kwargs['start_datetime'] = pd.Timestamp(kwargs.pop('start_datetime')).to_pydatetime()
        elif 'start_lst' in kwarg_keys:
            new_kwargs['start_lst'] = u.Quantity(kwargs.pop('start_lst'), u.hourangle).value
        elif 'start_elev' in kwarg_keys:
            new_kwargs['start_elev'] = u.Quantity(kwargs.pop('start_elev'), u.deg).value
            new_kwargs['moving_up'] = kwargs.pop('moving_up', True)

        if kwargs:
            raise TypeError(f'Unrecognized observation parameters: {kwargs.keys()}')

        return new_kwargs

    def _clean_param_telescope_data(self, need_lst, **kwargs):
        kwarg_keys = kwargs.keys()
        new_kwargs = dict()

        # required
        new_kwargs['lat'] = u.Quantity(kwargs.pop('lat', FYST_LOC.lat), u.deg).value

        if need_lst:

            # choose between
            if np.count_nonzero( ['start_ra' in kwarg_keys, 'start_datetime' in kwarg_keys, 'start_lst' in kwarg_keys] ) != 1:
                raise TypeError('need one (and only one) of start_ra, start_datetime, or start_lst')

            if 'start_ra' in kwarg_keys:
                new_kwargs['start_ra'] = u.Quantity(kwargs.pop('start_ra'), u.deg).value
            elif 'start_datetime' in kwarg_keys:
                new_kwargs['start_datetime'] = pd.Timestamp(kwargs.pop('start_datetime')).to_pydatetime()
            elif 'start_lst' in kwarg_keys:
                new_kwargs['start_lst'] = u.Quantity(kwargs.pop('start_lst'), u.hourangle).value

        if kwargs:
            raise TypeError(f'Unrecognized observation parameters: {kwargs.keys()}')

        return new_kwargs

    def _from_sky_pattern(self, sky_pattern):

        if max(abs(sky_pattern.x_coord.value)) > 10:
            warnings.warn('This is a larger pattern and the conversion between x and y deltas and RA/DEC may be slightly off.')

        param = self.param

        # get alt/az
        start_dec = param['start_dec'].to(u.rad).value
        hour_angle = self.lst - (sky_pattern.x_coord/cos(start_dec) + param['start_ra']) # FIXME fine for small regions, but consider checking out https://docs.astropy.org/en/stable/coordinates/matchsep.html for larger regions

        hour_angle_rad = hour_angle.to(u.rad).value
        dec_rad = (sky_pattern.y_coord + param['start_dec']).to(u.rad).value
        lat_rad = param['lat'].to(u.rad).value

        alt_rad = np.arcsin( np.sin(dec_rad)*sin(lat_rad) + np.cos(dec_rad)*cos(lat_rad)*np.cos(hour_angle_rad) )

        cos_az_rad = (np.sin(dec_rad) - np.sin(alt_rad)*sin(lat_rad)) / (np.cos(alt_rad)*cos(lat_rad))
        cos_az_rad[cos_az_rad > 1] = 1
        cos_az_rad[cos_az_rad < -1] = -1

        az_rad = np.arccos( cos_az_rad )
        mask = np.sin(hour_angle_rad) > 0 
        az_rad[mask] = 2*pi - az_rad[mask]

        return np.degrees(az_rad), np.degrees(alt_rad)

    def _get_lst(self, sky_pattern=None):

        if not sky_pattern is None:
            extra_ra_offset = sky_pattern.x_coord[0]
            extra_dec_offset = sky_pattern.y_coord[0]
        else:
            extra_ra_offset = 0*u.deg
            extra_dec_offset = 0*u.deg

        param = self.param

        # given a starting datetime
        if 'start_datetime' in param.keys():
            start_datetime = Time(param['start_datetime'], location=(param['lat'], 0*u.deg))
            start_lst = start_datetime.sidereal_time('apparent')

        # given a starting hourangle
        elif 'start_hrang' in param.keys() and not sky_pattern is None:
            start_lst = param['start_hrang'] + extra_ra_offset + param['start_ra']
        
        # given a starting lst
        elif 'start_lst' in param.keys():
            start_lst = param['start_lst']

        # given a starting elevation
        elif 'start_elev' in param.keys() and not sky_pattern is None:

            # determine possible hour angles
            alt_rad = param['start_elev'].to(u.rad).value
            dec_rad = (param['start_dec'] + extra_dec_offset).to(u.rad).value
            lat_rad = param['lat'].to(u.rad).value

            try:
                start_hrang_rad = math.acos((sin(alt_rad) - sin(dec_rad)*sin(lat_rad)) / (cos(dec_rad)*cos(lat_rad)))
            except ValueError:
                max_el = math.floor(math.degrees(math.asin(cos(0)*cos(dec_rad)*cos(lat_rad) + sin(dec_rad)*sin(lat_rad))))
                min_el = math.ceil(math.degrees(math.asin(cos(pi)*cos(dec_rad)*cos(lat_rad) + sin(dec_rad)*sin(lat_rad))))
                raise ValueError(f'Elevation = {param["start_elev"]} is not possible at provided ra, dec, and latitude. Min elevation is {min_el} and max elevation is {max_el} deg.')

            # choose hour angle
            if param['moving_up']:
                start_hrang_rad = -start_hrang_rad

            # starting sidereal time
            start_lst = start_hrang_rad*u.rad + extra_ra_offset + param['start_ra']
        
        elif 'start_ra' in param.keys() and sky_pattern is None:
            lat_rad = param['lat'].to(u.rad).value
            alt_rad = self.alt_coord[0].to(u.rad).value
            az_rad = self.az_coord[0].to(u.rad).value

            dec_rad = math.asin( sin(lat_rad)*sin(alt_rad) + cos(lat_rad)*cos(alt_rad)*cos(az_rad) )
            hrang_rad = math.acos( (sin(alt_rad) - sin(dec_rad)*sin(lat_rad)) / (cos(dec_rad)*cos(lat_rad)) )

            if sin(az_rad) > 0:
                hrang_rad = 2*pi - hrang_rad

            start_lst = hrang_rad*u.rad + param['start_ra']

        # find sidereal time
        SIDEREAL_TO_UT1 = 1.002737909350795
        return (self.time_offset.value/3600*SIDEREAL_TO_UT1*u.hourangle + start_lst).to(u.hourangle).value

    # TRANSFORMATIONS

    def _norm_angle(self, az):
        # normalize azimuth values so that:
        # 1. starting azimuth is between 0 and 360
        # 2. azmiuth values are between start-180 to start+180 (centered around the start)
        # formula = ((value - low) % diff) + low 

        lowest_az = az[0]%360 - 180
        return (az - lowest_az)%(360) + lowest_az 

    def _true_module_loc(self, module):
        # Gets the (dist, theta) of the module from the boresight (not necessarily central tube)

        if module == 'boresight':
            return 0, 0

        # passed by module identifier or instrument slot name
        if isinstance(module, str):
            try:
                return self.instrument.get_module_location(module, from_boresight=True).value
            except AttributeError as e:
                raise AttributeError(f'"instrument" is of type {type(self.instrument)}')
            except ValueError:
                try:
                    return self.instrument.get_slot_location(module, from_boresight=True).value
                except KeyError:
                    raise ValueError(f'{module} is not an existing module name or instrument slot')
        else:
            return self.instrument.location_from_boresight(module[0], module[1]).value

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
                a0 = root_scalar(func, args=(a1), x0=guess, bracket=[-pi/2, pi/2], xtol=10**(-9)).root
                guess = a0
            except ValueError:
                print(f'nan value at {i}')
                alt0 = math.nan

            alt0[i]= a0
        
        if len(alt0[alt0 < 0]) > 0:
            warnings.warn('elevation has values below 0')

        # getting new azimuth
        #cos_diff_az0 = ( np.cos(alt0)*cos(dist) - np.sin(alt0)*sin(dist)*np.sin(theta + alt0) )/np.cos(alt1)
        cos_diff_az0 = (cos(dist) - np.sin(alt0)*np.sin(alt1))/(np.cos(alt0)*np.cos(alt1))
        cos_diff_az0[cos_diff_az0 > 1] = 1
        cos_diff_az0[cos_diff_az0 < -1] = -1
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

        return self._norm_angle(np.degrees(az0)), np.degrees(alt0)

    def _transform_from_boresight(self, az0, alt0, dist, theta):
        
        # convert everything into radians
        dist = math.radians(dist)
        theta = math.radians(theta)
        alt0 = np.radians(alt0)
        az0 = np.radians(az0)

        # find elevation offset
        alt1 = np.arcsin(np.sin(alt0)*cos(dist) + sin(dist)*np.cos(alt0)*np.sin(theta + alt0))
        if len(alt1[alt1 < 0]) > 0:
            warnings.warn('elevation has values below 0')

        # find azimuth offset
        sin_az1 = 1/np.cos(alt1) * ( np.cos(alt0)*np.sin(az0)*cos(dist) + np.cos(az0)*np.cos(theta + alt0)*sin(dist) - np.sin(alt0)*np.sin(az0)*sin(dist)*np.sin(theta + alt0) )
        cos_az1 = 1/np.cos(alt1) * ( np.cos(alt0)*np.cos(az0)*cos(dist) - np.sin(az0)*np.cos(theta + alt0)*sin(dist) - np.sin(alt0)*np.cos(az0)*sin(dist)*np.sin(theta + alt0) )
        az1 = np.arctan2(sin_az1, cos_az1)

        return self._norm_angle(np.degrees(az1)), np.degrees(alt1)

    # METHODS

    def view_module(self, module, includes_instr_offset=False):
        """
        Get a TelescopePattern object representing the AZ/EL coordinates of the
        provided module. 

        Parameters
        ------------------------
        module : str or two-tuple
            | 1. string indicating a module name in the instrument e.g. 'SFH'
            | 2. string indicating one of the default slots in the instrument e.g. 'c', 'i1'
            | 3. tuple of (distance, theta) indicating module's offset from the center of the instrument, default unit deg
        includes_instr_offset : bool, default False
            if "module" parameter is a tuple of (distance, theta), this includes the instrument offset

        Returns
        -------------------------
        TelescopePattern 
            A TelescopePattern object where the "boresight" is the path of the provided module.
        """
        
        if includes_instr_offset:
            assert(len(module) == 2)
            dist, theta = u.Quantity(module[0], u.deg).value, u.Quantity(module[1], u.deg).value
        else:
            dist, theta = self._true_module_loc(module)
            
        az1, alt1 = self._transform_from_boresight(self.az_coord.value, self.alt_coord.value, dist, theta)

        data = {'time_offset': self.time_offset.value, 'lst': self.lst.value, 'az_coord': az1, 'alt_coord': alt1}

        return TelescopePattern(data, lat=self.param['lat'])

    def get_sky_pattern(self) -> SkyPattern:
        """
        Returns
        -------------------------
        SkyPattern
            A SkyPattern object for the boresight.
        """

        start_dec = self.dec_coord[0].to(u.rad).value 

        data = {
            'time_offset': self.time_offset.value, 
            # FIXME fine for small regions, but consider checking out https://docs.astropy.org/en/stable/coordinates/matchsep.html for larger regions
            'x_coord': self.ra_coord.value*cos(start_dec) - self.ra_coord[0].value*cos(start_dec),
            'y_coord': self.dec_coord.value - self.dec_coord[0].value 
        }

        if max(abs(data['x_coord'])) > 10:
            warnings.warn('This is a larger pattern and the conversion between x and y deltas and RA/DEC may be slightly off.')

        return SkyPattern(data=data)

    def save_param(self, param_json=None):
        """
        Save observation parameters.

        Parameters
        ----------------------------
        path_or_buf : str, file handle, or None; default None
            File path or object, if `None` is provided the result is returned as a dictionary.
        
        Returns
        ----------------------
        None or dict
            If `path_or_buf` is `None`, returns the resulting json format as a dictionary. Otherwise returns `None`.
        """

        param_temp = self._param.copy()
        if 'start_datetime' in param_temp.keys():
            param_temp['start_datetime'] = param_temp['start_datetime'].strftime('%Y-%m-%d %H:%M:%S %z') 

        # save param_json
        if param_json is None:
            return param_temp
        else:
            with open(param_json, 'w') as f:
                json.dump(param_temp, f)

    def save_data(self, path_or_buf=None, columns='default'):
        """
        Save kinematics data of the boresight. 

        Parameters
        ----------------------
        path_or_buf : str, file handle or None; default None
            File path or object, if `None` is provided the result is returned as a dictionary.
        columns : sequence, str or [dict of str -> str/Unit/None]; default 'default'
            Columns to write. If `dict`, map column names to their desired unit and use `None` if you would like to use the standard.
            'default' for ['time_offset', 'lst', 'az_coord', 'alt_coord', 'az_vel', 'alt_vel'].
            'all' for ['time_offset', 'az_coord', 'alt_coord', 'az_vel', 'alt_vel', 'vel', 'az_acc', 'alt_acc', 'acc', 'az_jerk', 'alt_jerk', 'jerk', 'lst', 'hour_angle', 'para_angle', 'rot_angle', 'ra_coord', 'dec_coord'].
        
        Returns
        ----------------------
        None or [dict of str -> array]
            If `path_or_buf` is `None`, returns the data as a dictionary mapping column name to values. Otherwise returns `None`.

        Examples
        ---------------------
        >>> telescope_pattern.save_data('file.csv', columns={'time_offset': 'sec', 'alt_coord': 'arcsec', 'az_coord': None})
        """

        # replace str options
        if columns == 'default':
            columns = ['time_offset', 'lst', 'az_coord', 'alt_coord', 'az_vel', 'alt_vel']
        elif columns == 'all':
            columns = ['time_offset', 'az_coord', 'alt_coord', 'az_vel', 'alt_vel', 'vel', 'az_acc', 'alt_acc', 'acc', 'az_jerk', 'alt_jerk', 'jerk', 'lst', 'hour_angle', 'para_angle', 'rot_angle', 'ra_coord', 'dec_coord']
        
        data = pd.DataFrame()

        # generate required data
        if isinstance(columns, dict):
            for col, unit in columns.items():
                if not unit is None:
                    data[col] = getattr(self, col).to(unit).value
                else:
                    data[col] = getattr(self, col).value
        else:
            for col in columns:
                data[col] = getattr(self, col).value
        
        # returning
        if path_or_buf is None:
            return data.to_dict('list')
        else:
            data.to_csv(path_or_buf, index=False)

    # ATTRIBUTES
    
    @property
    def param(self):
        """dict: Parameters inputted by user."""
        return_param = dict()
        for p, val in self._param.items():
            return_param[p] = val if self._param_units[p] is u.dimensionless_unscaled else val*self._param_units[p]
        return return_param

    @property
    def instrument(self):
        """ Insturment: Insturment object."""
        return self._instrument

    @instrument.setter
    def instrument(self, value):
        self._instrument = value

    @property
    def scan_duration(self):
        """Quantity: Total scan duration."""
        return self.time_offset[-1] + self.sample_interval

    @property
    def sample_interval(self):
        """Quantity: Time between samples."""
        return self._sample_interval*self._stored_units['time_offset']

    # Other Motions/Time

    @property
    def time_offset(self):
        """Quantity array: Time offsets."""
        return self._data['time_offset'].to_numpy()*self._stored_units['time_offset']
    
    @property
    def lst(self):
        """Quantity array: Local sidereal time."""
        return self._data['lst'].to_numpy()*self._stored_units['lst']

    @property
    def ra_coord(self):
        """Quantity array: Right ascension."""
        return self._norm_angle((self.lst - self.hour_angle).to(u.deg).value)*u.deg

    @property
    def dec_coord(self):
        """Quantity array: Declination."""
        lat_rad = radians(self._param['lat'])
        alt_coord_rad = self.alt_coord.to(u.rad).value
        az_coord_rad = self.az_coord.to(u.rad).value

        dec_rad = np.arcsin( sin(lat_rad)*np.sin(alt_coord_rad) + cos(lat_rad)*np.cos(alt_coord_rad)*np.cos(az_coord_rad) )
        return np.degrees(dec_rad)*u.deg

    @property
    def hour_angle(self):
        """Quantity array: Hour angle."""
        lat_rad = radians(self._param['lat'])
        alt_coord_rad = self.alt_coord.to(u.rad).value
        az_coord_rad = self.az_coord.to(u.rad).value
        dec_rad = self.dec_coord.to(u.rad).value

        hrang_rad = np.arccos( (np.sin(alt_coord_rad) - np.sin(dec_rad)*sin(lat_rad)) / (np.cos(dec_rad)*cos(lat_rad)) )
        
        mask = np.sin(az_coord_rad) > 0
        hrang_rad[mask] = 2*pi - hrang_rad[mask]

        return (hrang_rad*u.rad).to(u.hourangle)

    @property
    def para_angle(self):
        """Quantity array: Parallactic angle."""
        dec_rad = self.dec_coord.to(u.rad).value
        hour_angle_rad = self.hour_angle.to(u.rad).value
        lat_rad = radians(self._param['lat'])

        para_angle_deg = np.degrees(np.arctan2( 
            np.sin(hour_angle_rad), 
            np.cos(dec_rad)*tan(lat_rad) - np.sin(dec_rad)*np.cos(hour_angle_rad) 
        ))
        return self._norm_angle(para_angle_deg)*u.deg

    @property
    def rot_angle(self):
        """Quantity array: Field rotation (elevation + parallactic angle)."""
        return self._norm_angle((self.para_angle + self.alt_coord).value)*u.deg

    # Azimuthal/Elevation Motion

    @property
    def az_coord(self):
        """Quantity array: Azimuth coordinates (in terms of East of North)."""
        return self._data['az_coord'].to_numpy()*self._stored_units['az_coord']
    
    @property
    def alt_coord(self):
        """Quantity array: Elevation coordinates."""
        return self._data['alt_coord'].to_numpy()*self._stored_units['alt_coord']

    @property
    def az_vel(self):
        """Quantity array: Azimuth velocity."""
        return _central_diff(self.az_coord.value, self.sample_interval.value)*(self._stored_units['az_coord']/self._stored_units['time_offset'])

    @property
    def alt_vel(self):
        """Quantity array: Elevation velocity."""
        return _central_diff(self.alt_coord.value, self.sample_interval.value)*(self._stored_units['alt_coord']/self._stored_units['time_offset'])

    @property
    def vel(self):
        """Quantity array: Total velocity."""
        return np.sqrt(self.az_vel**2 + self.alt_vel**2)

    @property
    def az_acc(self):
        """Quantity array: Azimuth acceleration."""
        return _central_diff(self.az_vel.value, self.sample_interval.value)*(self._stored_units['az_coord']/self._stored_units['time_offset']**2)

    @property
    def alt_acc(self):
        """Quantity array: Elevation acceleration."""
        return _central_diff(self.alt_vel.value, self.sample_interval.value)*(self._stored_units['alt_coord']/self._stored_units['time_offset']**2)
    
    @property
    def acc(self):
        """Quantity array: Total acceleration."""
        return np.sqrt(self.az_acc**2 + self.alt_acc**2)

    @property
    def az_jerk(self):
        """Quantity array: Azimuth jerk."""
        return _central_diff(self.az_acc.value, self.sample_interval.value)*(self._stored_units['az_coord']/self._stored_units['time_offset']**3)
    
    @property
    def alt_jerk(self):
        """Quantity array: Elevation jerk."""
        return _central_diff(self.alt_acc.value, self.sample_interval.value)*(self._stored_units['alt_coord']/self._stored_units['time_offset']**3)
    
    @property
    def jerk(self):
        """Quantity array: Total jerk."""
        return np.sqrt(self.az_jerk**2 + self.alt_jerk**2)
