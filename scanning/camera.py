import math
from math import cos, sin, radians, degrees, sqrt
import json
import warnings
from functools import wraps

import numpy as np
import pandas as pd
from scipy.constants import speed_of_light
import astropy.units as u

#####################
#   CAMERA MODULE
#####################

class Module():
    """
    Class for a single camera module (e.g. EoRSpec, CMBPol, SFH). Each module 
    consists of three wafers/detector arrays. Each wafter contains three rhombuses. 
    """

    _stored_units = {'ang_res': u.deg, 'x': u.deg, 'y': u.deg, 'pol': u.deg, 'freq': u.Hz*10**9}

    # other attributes (for development):
    # _stored_units
    # _data: x, y, pol, pixel_num, rhombus, wafer
    # _ang_res
    # _freq

    # INITIALIZATION  

    def __init__(self, data=None, units=None, **kwargs) -> None:
        """
        Create a camera module either through:
            | option1 : Module(data, units)
            | option2 : Module(freq=, F_lambda=)
            | option3 : Module(wavelength=, F_lambda=)

        Parameters
        --------------------
        data : str, DataFrame or [dict of str -> sequence]
            If `str`, a file path to a csv file. If `dict` or `DataFrame`, column names map to their values. 
            Must have columns: x, y. Recommended to have columns: pol, rhombus, and wafer. 
        units : [dict of str -> str or Unit] or None; default None
            Mapping columns in `data` with their units. All columns do not need to be mapped.
            If not provided, all applicable units are assumed to be in degrees. 
        
        Keyword Arguments
        -------------------
        freq : float, Quantity or str; default unit GHz
            Center of frequency band. Must be positive. 
            If each wafer is different (such as EoRSpec), pass a three-length list like [freq1, freq2, freq3].
        wavelength : float, Quantity or str; default unit micron
            Intended wavelength of light. Must be positive. 
            If each wafer is different (such as EoRSpec), pass a three-length list like [wavelength1, wavelength2, wavelength3].
        F_lambda : float; default 1.2
            Factor for spacing between individual detectors.

        Raises
        -------------------
        ValueError
            `data` could not be properly interpreted.
        TypeError
            Missing parameters or unneccesary keywords.

        Examples
        --------------------------
        >>> Module('file.csv', units={'x': 'arcsec'}) # specify x is in arcsec
        >>> Module(freq=280, F_lambda=1.2)
        >>> Module(wavelength='350 micron')
        """

        # PASS IN DATA

        if not data is None:
            try:
                if isinstance(data, str):
                    self._data = pd.read_csv(data, index_col=False)
                else:
                    self._data = pd.DataFrame(data)
            except (ValueError, TypeError):
                raise ValueError('"data" could not be properly interpreted')

            # check for certain columns inside
            for col in ['pol', 'rhombus', 'wafer']:
                if not col in self._data.columns:
                    warnings.warn(f'column {col} not in data, marking all column values for {col} as 0')
                    self._data[col] = 0

            if not 'pixel_num' in self._data.columns:
                self._data['pixel_num'] = self._data.index
            
            data_columns = ['x', 'y', 'pixel_num', 'pol', 'rhombus', 'wafer']
            self._data = self._data[data_columns]

            # convert to specified units
            if not units is None:
                for col, unit in units.items():
                    self._data[col] = self._data[col]*u.Unit(unit).to(self._stored_units[col])

            self._data = self._data.astype({'pixel_num': int, 'pol': np.int16, 'rhombus': np.uint8, 'wafer': np.uint8})

            # find angular resolution and freq
            self._ang_res = self._find_ang_res()
            self._freq = self._find_freq()

        # GENERATE MODULE

        else:

            # check F_lambda
            F_lambda = kwargs.pop('F_lambda', 1.2)

            # get freq
            if 'freq' in kwargs.keys():
                freq = u.Quantity(kwargs.pop('freq'), self._stored_units['freq']).value
            elif 'wavelength' in kwargs.keys():
                wavelength = u.Quantity(kwargs.pop('wavelength'), u.micron).to(u.m).value
                freq = speed_of_light/wavelength/10**9
            else:
                raise TypeError('cannot create Module without one of file, freq, or wavelength')

            if kwargs:
                raise TypeError(f'unneccary keywords passed: {kwargs}')

            self._freq = freq
            self._ang_res, self._data = self._generate_module(freq, F_lambda)        
  
    def _find_ang_res(self):
        ang_res = []
        for wafer in np.unique(self.wafer):
            temp_dict = self._data[self.wafer == wafer].iloc[0:2].loc[:, ['x', 'y']].to_dict('list')
            ang_res.append( math.sqrt( (temp_dict['x'][0] - temp_dict['x'][1])**2 + (temp_dict['y'][0] - temp_dict['y'][1])**2 ) )
        
        if np.allclose([ang_res[0]]*len(ang_res), ang_res):
            return ang_res[0]
        else:
            return ang_res

    def _find_freq(self):
        freq = []
        for wafer in np.unique(self.wafer):
            num_wafer_pixels = np.count_nonzero(self.wafer == wafer)
            freq0 = 280*math.sqrt(num_wafer_pixels/3)/24
            freq.append(freq0)
        
        if np.allclose([freq[0]]*len(freq), freq):
            return freq[0]
        else:
            return freq

    def _waferpixelpos(self, p, numrows, numcols, numrhombus, centeroffset):
        """Obtains pixel positions in a wafer, origin centered at wafer center"""

        theta_axes = 2*np.pi/numrhombus # 120 degree offset between the row and column axes of each rhombus
        numpixels = numrows*numcols*numrhombus # number of total pixels on a detector wafer
        pixelpos = np.zeros((numpixels, 4)) # array for storing all pixel x and y coordinates, polarizations, and rhombus
        count = 0 # counter for pixel assignment to rows in the pixelpos array
        rhombusaxisarr = np.arange(0, 2*np.pi, theta_axes) # the 3 rhombus axes angles when drawn from the center of detector

        # Calculate pixel positions individually in a nested loop
        # Note there are more efficient ways of doing this, but nested for loops is probably most flexible for changes
        for i, pol in zip( range(numrhombus), ((45, 0), (-45, 60), (45, 30)) ):

            # convert polarizations to deg
            pm = pol[0]
            pc = pol[1]

            # For each rhombus, determine the angle that it is rotated about the origin,
            # and the position of the pixel nearest to the origin
            rhombusaxis = rhombusaxisarr[i]
            x0 = centeroffset*np.cos(rhombusaxis)
            y0 = centeroffset*np.sin(rhombusaxis)

            # Inside each rhombus iterate through each pixel by deterimining the row position,
            # and then counting 24 pixels by column along each row
            for row in np.arange(numrows):
                xrowstart = x0 + row * p*np.cos(rhombusaxis-theta_axes/2)
                yrowstart = y0 + row * p*np.sin(rhombusaxis-theta_axes/2)
                for col in np.arange(numcols):
                    x = xrowstart + col * p*np.cos(rhombusaxis+theta_axes/2)
                    y = yrowstart + col * p*np.sin(rhombusaxis+theta_axes/2)
                    pixelpos[count, :] = x, y, (count%2)*pm + pc, i
                    count = count + 1

        return pixelpos

    def _generate_module(self, center_freq, F_lambda=1.2):
        """
        Generates the pixel positions of a module centered at a particular frequency, based 
        off of code from the eor_spec_models package. The numerical values in the code are
        based off of the parameters for a detector array centered at 280 GHz. 
        """

        # --- SETUP ---

        # λ/D in rad (D = 5.8m, diameter of the telescope)

        tel_d = 5.8

        if isinstance(center_freq, int) or isinstance(center_freq, float):
            center1 = center2 = center3 = center_freq
            ang_res = ang_res1 = ang_res2 = ang_res3 = F_lambda*math.degrees((speed_of_light)/(center_freq*10**9)/tel_d)
        elif len(center_freq) == 3:
            center1 = center_freq[0]
            center2 = center_freq[1]
            center3 = center_freq[2]

            ang_res1 = F_lambda*math.degrees((speed_of_light)/(center1*10**9)/tel_d)
            ang_res2 = F_lambda*math.degrees((speed_of_light)/(center2*10**9)/tel_d)
            ang_res3 = F_lambda*math.degrees((speed_of_light)/(center3*10**9)/tel_d)
            ang_res = [ang_res1, ang_res2, ang_res3]
        else:
            raise ValueError(f'freq {center_freq} is invalid')

        # Each detector wafer is 72mm from the center of the focal plane
        # Let wafer 1 be shifted in +x by 72mm,
        # and wafer 2 and 3 be rotated to 120, 240 degrees from the x-axis respectively

        waferoffset = 72*10**-3 # distance from focal plane center to wafer center
        waferangles = [0, 2*np.pi/3, 4*np.pi/3] # the wafer angle offset from the x-axis

        # Wafer 1 
        ratio1 = 280/center1
        p1 = 2.75*10**-3 * ratio1 # pixel spacing (pitch)
        numrows1 = int(24 /ratio1) # number of rows in a rhombus
        numcols1 = int(24 /ratio1) # number of columns in a rhombus
        numrhombus1 = 3 # number of rhombuses on a detector wafer
        pixeloffset1 = 1.5*2.75*10**-3 # distance of nearest pixel from center of the wafer
        numpixels1 = numrows1*numcols1*numrhombus1 # number of total pixels on a detector wafer

        # Wafer 2
        ratio2 = 280/center2
        p2 = 2.75*10**-3 *ratio2 
        numrows2 = int(24 /ratio2) 
        numcols2 = int(24 /ratio2) 
        numrhombus2 = 3 
        pixeloffset2 = 1.5*2.75*10**-3 
        numpixels2 = numrows2*numcols2*numrhombus2

        # Wafer 3
        ratio3 = 280/center3
        p3 = 2.75*10**-3 *ratio3 
        numrows3 = int(24 /ratio3) 
        numcols3 = int(24 /ratio3) 
        numrhombus3 = 3 
        pixeloffset3 = 1.5*2.75*10**-3 
        numpixels3 = numrows3*numcols3*numrhombus3

        # --- DETECTOR PLANE PIXEL POSITIONS ---

        # Obtain pixel coordinates for a wafer with above parameters
        pixelcoords1 = self._waferpixelpos(p1,numrows1,numcols1,numrhombus1,pixeloffset1)
        pixelcoords2 = self._waferpixelpos(p2,numrows2,numcols2,numrhombus2,pixeloffset2)
        pixelcoords3 = self._waferpixelpos(p3,numrows3,numcols3,numrhombus3,pixeloffset3)
        
        ### Detector Array 1

        # wafer center coordinate, relative to center of the detector
        offsetarray1 = [waferoffset*np.cos(waferangles[0]), waferoffset*np.sin(waferangles[0])]

        # pixel coordinates, relative to center of the detector
        pixelcoords1[:,0] = pixelcoords1[:,0] + offsetarray1[0]
        pixelcoords1[:,1] = pixelcoords1[:,1] + offsetarray1[1]

        ### Detector Array 2
        offsetarray2 = [waferoffset*np.cos(waferangles[1]), waferoffset*np.sin(waferangles[1])]
        pixelcoords2[:,0] = pixelcoords2[:,0] + offsetarray2[0]
        pixelcoords2[:,1] = pixelcoords2[:,1] + offsetarray2[1]

        ### Detector Array 3
        offsetarray3 = [waferoffset*np.cos(waferangles[2]), waferoffset*np.sin(waferangles[2])]
        pixelcoords3[:,0] = pixelcoords3[:,0] + offsetarray3[0]
        pixelcoords3[:,1] = pixelcoords3[:,1] + offsetarray3[1]

        # --- CLEAN UP ---

        # turn to deg 
        pixelcoords1[: , 0] = pixelcoords1[: , 0]/p1*ang_res1
        pixelcoords2[: , 0] = pixelcoords2[: , 0]/p2*ang_res2
        pixelcoords3[: , 0] = pixelcoords3[: , 0]/p3*ang_res3

        pixelcoords1[: , 1] = pixelcoords1[: , 1]/p1*ang_res1
        pixelcoords2[: , 1] = pixelcoords2[: , 1]/p2*ang_res2
        pixelcoords3[: , 1] = pixelcoords3[: , 1]/p3*ang_res3

        # mark rhombuses
        pixelcoords2[:, 3] = pixelcoords2[:, 3] + 3
        pixelcoords3[:, 3] = pixelcoords3[:, 3] + 6

        # mark wafers
        pixelcoords1 = np.hstack( (pixelcoords1, [[0]]*numpixels1) )
        pixelcoords2 = np.hstack( (pixelcoords2, [[1]]*numpixels2) )
        pixelcoords3 = np.hstack( (pixelcoords3, [[2]]*numpixels3) )

        # save to data frame 
        data = pd.DataFrame(
            np.append(np.append(pixelcoords1, pixelcoords2, axis=0), pixelcoords3, axis=0), 
            columns=['x', 'y', 'pol', 'rhombus', 'wafer']
        ).astype({'pol': np.int16, 'rhombus': np.uint8, 'wafer': np.uint8})
        data['pixel_num'] = data.index

        return ang_res, data

    # METHODS

    def save_data(self, path_or_buf=None, columns='all'):
        """
        Write Module object to csv file.

        Parameters
        ----------------------
        path_or_buf : str, file handle or None; default None
            File path or object, if `None` is provided the result is returned as a dictionary.
        columns : sequence, str or [dict of str -> str/Unit/None]; default 'all'
            Columns to write. If `dict`, map column names to their desired unit and use `None` if column is unit-less.
            'all' is ['x', 'y', 'pol', 'pixel_num', 'rhombus', 'wafer']

        Returns
        ----------------------
        None or [dict of str -> array]
            If `path_or_buf` is `None`, returns the data as a dictionary mapping column name to values. Otherwise returns `None`.

        
        Examples
        ---------------------
        >>> # saves x, y, and rhombus
        >>> module.save_data(path_or_buf='file.csv', columns={'x': 'arcsec', 'y': 'rad', 'rhombus': None})

        """

        if columns == 'all':
            columns = ['x', 'y', 'pol', 'pixel_num', 'rhombus', 'wafer']

        # unit conversions

        if isinstance(columns, dict):
            data = pd.DataFrame()
            for col, unit in columns.items():
                if not unit is None:
                    data[col] = getattr(self, col).to(unit).value
                else:
                    data[col] = getattr(self, col)
        else:
            data = self._data[columns]

        # returning
        if path_or_buf is None:
            return data.to_dict('list')
        else:
            data.to_csv(path_or_buf, index=False)

    # ATTRIBUTES 

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
    def F_lambda(self):
        """
        Quantity: F lambda.
        """
        tel_d = 5.8
        return self._ang_res/math.degrees(speed_of_light/self._freq.to(u.Hz)/tel_d)

    @property
    def ang_res(self):
        """
        Quantity or Quantity array: Angular resolution. If multiple frequencies or wavelengths were provided, this will be a three-length sequence.  
        """
        return self._ang_res*self._stored_units['ang_res']

    @property
    def freq(self):
        """
        Quantity or Quantity array: Center of frequency band. If multiple frequencies or wavelengths were provided, this will be a three-length sequence.  
        """
        return self._freq*self._stored_units['freq']

    @property
    def x(self):
        """
        Quantity array: x offset of detector pixels.
        """
        return self._data['x'].to_numpy()*self._stored_units['x']
    
    @property
    def y(self):
        """
        Quantity array: y offset of detector pixels
        """
        return self._data['y'].to_numpy()*self._stored_units['y']
    
    @property
    def pol(self):
        """
        Quantity array: Polarization geometry. 
        """
        return self._data['pol'].to_numpy()*self._stored_units['pol']

    @property
    def pixel_num(self):
        """
        int array: Pixel mumber.
        """
        return self._data['pixel_num'].to_numpy()
    
    @property
    def rhombus(self):
        """
        int array: Rhombus number.
        """
        return self._data['rhombus'].to_numpy()

    @property
    def wafer(self):
        """
        int array: Wafer number.
        """
        return self._data['wafer'].to_numpy()
    

# some standard modules 
CMBPol = Module(freq=350)
SFH = Module(freq=860)
EoRSpec = Module(freq=[262.5, 262.5, 367.5])
Mod280 = Module(freq=280)

#######################
#     INSTRUMENT
#######################

class Instrument():
    """ A configurable instrument that holds modules. """

    # other attributes (for development)
    # _modules: mod_rot, distance, theta, module
    # _slots
    # _instr_rot
    # _instr_offset

    _slots = dict()

    def __init__(self, data=None, instr_offset=(0, 0), instr_rot=0) -> None:
        """
        Initialize a filled Instrument:
            option 1: Instrument(data) 
        or an empty Intrument:
            option 2: Instrument(instr_offset, instr_rot)

        Parameters
        -----------------------------
        data : str or dict
            File path to json file or dict object. Overwrites `instr_offset` and `instr_rot`. Applicable values are in degrees unit. 
        instr_offset : (float/Quantity/str, float/Quantity/str); default (0, 0), default unit deg
            Offset of the instrument from the boresight.
        instr_rot : float, Quantity or str; default 0, default unit deg
            CCW rotation of the instrument.

        Raises 
        -------------------------------
        ValueError
            could not parse "data"

        Examples
        ------------------------------
        >>> import astropy.units as u
        >>> Instrument(inst_offset=(0, '100 arcsec'), instr_rot=3.14*u.rad)
        >>> Instrument('file.json')
        """

        # config file is passed
        if not data is None:

            if isinstance(data, str):
                with open(data, 'r') as f:
                    config = json.load(f)
            elif isinstance(data, dict):
                config = data
            else:
                raise TypeError('cannot parse "data"')

            try:
                # populate modules
                self._modules = dict()
                for identifier in config['modules'].keys():
                    self._modules[identifier] = {prop: config['modules'][identifier].pop(prop) for prop in ('dist', 'theta', 'mod_rot')}
                    self._modules[identifier]['module'] = Module(config['modules'][identifier])
            except (KeyError, ValueError):
                raise ValueError('could not parse "data"')
            
            self.instr_offset = config['instr_offset']
            self.instr_rot = config['instr_rot']

        # empty instrument
        else:
            self.instr_rot = instr_rot
            self.instr_offset = instr_offset
            
            self._modules = dict()
    
    def __repr__(self) -> str:
        instr_repr = f'instrument: offset {self.instr_offset}, rotation {self.instr_rot}\n------------------------------------'

        if len(self._modules) == 0:
            instr_repr += '\nempty'
        else:
            for identifier in self._modules.keys():
                instr_repr +=  "\n" + f"{identifier} \n (r, theta) = {(self._modules[identifier]['dist'], self._modules[identifier]['theta'])}, rotation = {self._modules[identifier]['mod_rot']}"
        
        return instr_repr

    def _check_identifier(func):
        @wraps(func)
        def wrapper(self, identifier, *args, **kwargs):
            if not identifier in self._modules.keys():
                raise ValueError(f'identifier {identifier} is not valid')
            return func(self, identifier, *args, **kwargs)
        return wrapper

    # CHANGING INSTRUMENT CONFIGURATION

    def add_module(self, module, location, mod_rot=0, identifier=None) -> None:
        """
        Add a module to the instrument.

        Parameters
        -------------------------
        module : Module or str
            A `Module` object or one of the default options ['CMBPol', 'SFH', 'EoRSpec', 'Mod280']
        location : (float/Quantity/str, float/Quantity/str) or str; default unit deg
            A `tuple` containing the (distance, theta) from the center of the instrument in polar coordinates.
            Or a `str` of one of the default slot options. 
        identifier : str or None
            Name of the module. If user chose a default `module` option and this is `None`, then `identifier` will be its corresponding name. 
        mod_rot : float/Quantity/str; default 0, default unit deg
            CCW rotation of the module.
        """

        # if a default option of module was chosen
        if isinstance(module, str):
            if identifier is None:
                identifier = module

            if module == 'CMBPol':
                module = CMBPol
            elif module == 'SFH':
                module = SFH
            elif module == 'EoRSpec':
                module = EoRSpec
            elif module == 'Mod280':
                module = Mod280
            else:
                raise ValueError(f'module {module} is not a valid option')

        elif not isinstance(identifier, str):
            raise ValueError('Identifier string must be passed')

        # if a default option for location was chosen
        if isinstance(location, str):
            location = self.slots[location].value
        else:
            location = u.Quantity(location, u.deg).value

        # change module rotation
        mod_rot = u.Quantity(mod_rot, u.deg).value
        
        # if identifier is already a module that's saved
        if identifier in self._modules.keys():
            warnings.warn(f'Module {identifier} already exists. Overwriting it...')

        self._modules[identifier] = {'module': module, 'dist': location[0], 'theta': location[1], 'mod_rot': mod_rot}

    @_check_identifier
    def change_module(self, identifier, new_location=None, new_mod_rot=None, new_identifier=None) -> None:
        """
        Change a module in the instrument.

        Parameters
        -------------------------
        identifier : str
            Name of the module to change. 
        new_location : (float/Quantity/str, float/Quantity/str) or str; default unit deg, optional
            A `tuple` containing the (distance, theta) from the center of the instrument in polar coordinates.
            Or a `str` of one of the default slot options. 
        new_mod_rot : float/Quantity/str; default 0, default unit deg, optional
            CCW rotation of the module.
        new_identifier : str, optional, optional
            New name of the module. 
        
        Raises
        ---------------------------
        ValueError
            "identifier" is not valid
        """
    
        if new_identifier is None and new_location is None and new_mod_rot is None:
            warnings.warn(f'Nothing has changed for {identifier}.')
            return

        # rename identifier
        if not new_identifier is None:
            self._modules[new_identifier] = self._modules.pop(identifier)
            identifier = new_identifier

        # change location
        if not new_location is None:
            if isinstance(new_location, str):
                new_location = self.slots[new_location].value
            else:
                if new_location[0] is None:
                    new_location[0] = self._modules[identifier]['dist']
                if new_location[1] is None:
                    new_location[1] = self._modules[identifier]['theta']
                new_location = u.Quantity(new_location, u.deg).value
                
            self._modules[identifier]['dist'] = new_location[0]
            self._modules[identifier]['theta'] = new_location[1]

        # change module rotation
        if not new_mod_rot is None:
            self._modules[identifier]['mod_rot'] = u.Quantity(new_mod_rot, u.deg).value

    @_check_identifier
    def delete_module(self, identifier) -> None:
        """
        Delete a module in the instrument. 

        Parameters
        -------------------------
        identifier : str
            Name of the module. 

        Raises
        ---------------------------
        ValueError
            "identifier" is not valid
        """
        self._modules.pop(identifier)

    # GETTERS 

    def save_data(self, path_or_buf=None):
        """
        Saves as a dictionary like in a json format.

        Parameters
        -------------
        path_or_buf : str or file handle, default None
            File path or object, if `None` is provided the result is returned as a `dict`.

        Returns
        ----------------------
        None or dict
            If `path_or_buf` is `None`, returns the resulting json format as a dictionary. Otherwise returns `None`.
        """

        # organize the configuration 
        config = {'instr_offset': list(self.instr_offset.value), 'instr_rot': self.instr_rot.value, 'modules': dict()}

        for identifier in self._modules.keys():
            config['modules'][identifier] = {
                'dist': self._modules[identifier]['dist'], 
                'theta': self._modules[identifier]['theta'],
                'mod_rot': self._modules[identifier]['mod_rot'],
                'x': self._modules[identifier]['module'].x.value.tolist(),
                'y': self._modules[identifier]['module'].y.value.tolist(),
                'pol': self._modules[identifier]['module'].pol.value.tolist(),
                'pixel_num': self._modules[identifier]['module'].pixel_num.tolist(),
                'rhombus': self._modules[identifier]['module'].rhombus.tolist(),
                'wafer': self._modules[identifier]['module'].wafer.tolist()
            }
        
        # push configuration 
        if path_or_buf is None:
            return config
        else:
            with open(path_or_buf, 'w') as f:
                json.dump(config, f)
    
    @_check_identifier
    def get_module(self, identifier, with_mod_rot=False, with_instr_rot=False, with_instr_offset=False, with_mod_offset=False) -> Module:
        """        
        Parameters
        -------------------------
        identifier : str
            Name of the module. 
        with_mod_rot, with_instr_rot, with_instr_offset, with_mod_offset : bool; default False
            Whether to apply instrument and module rotation and offset to the returned `Module`. 
        
        Returns
        --------------------------
        Module 
            `Module` object. 
        
        Raises
        ---------------------------
        ValueError
            "identifier" is not valid
        """

        if not np.any([with_mod_rot, with_instr_rot, with_instr_offset, with_mod_offset]):
            return self._modules[identifier]['module']
        else:
            mod_rot_rad = self.get_module_rot(identifier).to(u.rad).value if with_mod_rot else 0
            instr_rot_rad = self.instr_rot.to(u.rad).value if with_instr_rot else 0

            mod_dist = self.get_module_dist(identifier).value if with_mod_offset else 0
            mod_theta_rad = self.get_module_theta(identifier).to(u.rad).value if with_mod_offset else 0
            mod_x = mod_dist*cos(mod_theta_rad)
            mod_y = mod_dist*sin(mod_theta_rad)

            instr_offset = self.instr_offset.value if with_instr_offset else (0, 0)

            data = self._modules[identifier]['module'].save_data()
            x = np.array(data['x'])
            y = np.array(data['y'])

            data['x'] = x*cos(mod_rot_rad + instr_rot_rad) - y*sin(mod_rot_rad + instr_rot_rad) + mod_x + instr_offset[0]
            data['y'] = x*sin(mod_rot_rad + instr_rot_rad) + y*cos(mod_rot_rad + instr_rot_rad) + mod_y + instr_offset[1]

            return Module(data)

    @_check_identifier
    def get_module_dist(self, identifier, from_boresight=False):
        """
        Parameters
        -------------------------
        identifier : str
            Name of the module. 
        from_boresight : bool; default False
            From the boresight (`True`) or from the center of the instrument (`False`). 

        Returns
        -------------------------
        Quantity
            Distance away of the module in polar coordinates. 

        Raises
        ---------------------------
        ValueError
            "identifier" is not valid
        """
        return self.get_module_location(identifier, from_boresight)[0]

    @_check_identifier
    def get_module_theta(self, identifier, from_boresight=False):
        """
        Parameters
        -------------------------
        identifier : str
            Name of the module. 
        from_boresight : bool; default False
            From the boresight (`True`) or from the center of the instrument (`False`). 

        Returns
        -------------------------
        Quantity 
            Angle of the module in polar coordinates. 
        
        Raises
        ---------------------------
        ValueError
            "identifier" is not valid
        """
        return self.get_module_location(identifier, from_boresight)[1]
    
    @_check_identifier
    def get_module_rot(self, identifier, with_instr_rot=False):
        """
        Get rotation of module from the center of the instrument.

        Parameters
        -------------------------
        identifier : str
            Name of the module. 
        with_instr_rot : bool; default False
            Whether to include instrument rotation. 

        Returns
        --------------------------
        Quantity
            Rotation of the module 

        Raises
        ---------------------------
        ValueError
            "identifier" is not valid
        """
        if with_instr_rot:
            return self._modules[identifier]['mod_rot']*u.deg + self.instr_rot
        else:
            return self._modules[identifier]['mod_rot']*u.deg

    @_check_identifier
    def get_module_location(self, identifier, from_boresight=False, polar=True):
        """
        Parameters
        -------------------------
        identifier : str
            Name of the module. 
        from_boresight : bool; default False
            From the boresight (`True`) or from the center of the instrument (`False`). 
        polar : bool; default True
            In polar coordinates (`True`) or in cartesian coordinates (`False`)

        Returns
        ---------------------------
        Quantity two-tuple
            (distance, theta) location of module.

        Raises
        ---------------------------
        ValueError
            "identifier" is not valid
        """

        # get module location from center of instrument in x and y
        dist = self._modules[identifier]['dist']
        theta = self._modules[identifier]['theta']

        if from_boresight:
            theta_rad = radians(theta)
            mod_x = dist*cos(theta_rad)
            mod_y = dist*sin(theta_rad)

            # get rotation and offset
            instr_x = self.instr_offset[0].value
            instr_y = self.instr_offset[1].value
            instr_rot_rad = self.instr_rot.to(u.rad).value
            mod_rot_rad = radians(self._modules[identifier]['mod_rot'])

            # get module location from boresight in x and y
            new_mod_x = mod_x*cos(instr_rot_rad + mod_rot_rad) - mod_y*sin(instr_rot_rad + mod_rot_rad) + instr_x
            new_mod_y = mod_x*sin(instr_rot_rad + mod_rot_rad) + mod_y*cos(instr_rot_rad + mod_rot_rad) + instr_y

            new_dist = sqrt(new_mod_x**2 + new_mod_y**2)
            new_theta = degrees(math.atan2(new_mod_y, new_mod_x))

            dist = new_dist 
            theta = new_theta

        if polar:
            return (dist, theta)*u.deg
        else:
            return (dist*cos(radians(theta)), dist*sin(radians(theta)))*u.deg

    def get_slot_location(self, slot_name, from_boresight=False, polar=True):
        """
        Parameters
        -------------------------
        slot_name : str
            Name of slot. 
        from_boresight : bool; default False
            From the boresight (`True`) or from the center of the instrument (`False`). 
        polar : bool; default True
            In polar coordinates (`True`) or in cartesian coordinates (`False`)

        Returns
        ---------------------------
        Quantity two-tuple
            (distance, theta) location of slot.

        Raises
        ---------------------------
        ValueError
            "slot_name" is not valid
        """

        # get slot location from center of instrument in x and y
        try:
            dist = self.slots[slot_name][0].value
            theta = self.slots[slot_name][1].value
        except KeyError:
            raise ValueError(f'"slot_name" {slot_name} is not valid')

        if from_boresight:
            return self.location_from_boresight(dist, theta, polar)

        if polar:
            return (dist, theta)*u.deg
        else:
            return (dist*cos(radians(theta)), dist*sin(radians(theta)))*u.deg

    def location_from_boresight(self, dist, theta, polar=True):
        """
        Given a location relative to the center of the instrument, find the 
        location relative to the boresight. 

        Parameters
        -------------------
        dist : float, Quantity, or str; default unit deg
            Distance away from the center of the instrument.
        theta : float, Quantity, or str; default unit deg
            Angle away from the center of the instrument in polar coordinates. 
        polar : bool; default True
            In polar coordinates (`True`) or in cartesian coordinates (`False`)

        Returns
        ---------------------------
        Quantity two-tuple
            (distance, theta) location of slot. 
        
        """

        dist = u.Quantity(dist, u.deg).value
        theta_rad = u.Quantity(theta, u.deg).to(u.rad).value

        instr_x = self.instr_offset[0].value
        instr_y = self.instr_offset[1].value
        instr_rot_rad = self.instr_rot.to(u.rad).value

        # get true module location in terms of x and y
        orignal_x = dist*cos(theta_rad)
        original_y = dist*sin(theta_rad)

        x_offset = orignal_x*cos(instr_rot_rad) - original_y*sin(instr_rot_rad) + instr_x
        y_offset = orignal_x*sin(instr_rot_rad) + original_y*cos(instr_rot_rad) + instr_y

        new_dist = sqrt(x_offset**2 + y_offset**2)
        new_theta = math.degrees(math.atan2(y_offset, x_offset))  

        if polar:
            return (new_dist, new_theta)*u.deg
        else:
            return (new_dist*cos(radians(new_theta)), new_dist*sin(radians(new_theta)))*u.deg

    # ATTRIBUTES 

    @property
    def instr_offset(self):
        """Quantity two-tuple: The x and y offsets of the center of the instrument from the boresight. """
        return self._instr_offset*u.deg
    
    @property
    def instr_rot(self):
        """Quantity: The CCW rotation of the instrument."""
        return self._instr_rot*u.deg
    
    @property
    def slots(self):
        """dict of str -> Quantity two-tuple: Map of slot name to their (distance, theta) from the center of the instrument."""
        return {slot_name: slot_loc*u.deg for slot_name, slot_loc in self._slots.items()}

    @property
    def modules(self):
        """list of str: list of module identifiers inside the instrument."""
        return [module_name for module_name in self._modules.keys()]

    # SETTERS

    @instr_offset.setter
    def instr_offset(self, value):
        self._instr_offset = (u.Quantity(value[0], u.deg).value, u.Quantity(value[1], u.deg).value)

    @instr_rot.setter
    def instr_rot(self, value):
        self._instr_rot = u.Quantity(value, u.deg).value


class ModCam(Instrument): 
    _slots = {'c': (0, 0)}

class PrimeCam(Instrument):

    # default configuration of optics tubes at 0 deg elevation 
    # in terms of (radius from center [deg], angle [deg])
    _default_ir = 1.78
    _slots = {
        'c': (0, 0), 
        'i1': (_default_ir, -90), 'i2': (_default_ir, -30), 'i3': (_default_ir, 30), 'i4': (_default_ir, 90), 'i5': (_default_ir, 150), 'i6': (_default_ir, -150),
    }
    
