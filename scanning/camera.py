# imports
import math
from math import sin, cos
import numpy as np
import pandas as pd
from scipy.constants import speed_of_light

import json
import warnings

import astropy.units as u

""" - handling units (storing and checking), checking parameters type/config, error handling
- if polarization is in alternate direction
- Module csv file is just pixel positions and does not include pol

- more flexible file parsing (e.g. allow user to specify units, data is in dist not angle-like units)
"""

def param_type_checker(param, param_name, valid_types):
    if not isinstance(param, valid_types):
        raise TypeError(f'argument {param_name} has value {param} but must be of type {valid_types}')

#####################
#   CAMERA MODULE
#####################

class Module():
    """
    Class for a single camera module (e.g. EoRSpec, CMBPol, SFH). Each module 
    consists of three wafers/detector arrays. Each wafter contains three rhombuses. 
    
    Attributes
    ------------------
    data : pd.DataFrame
        x (deg), y (deg), pol (deg int), rhombus (int), wafer (int)
    prop : dict
        containing frequency and F_Lambda 
    """

    def __init__(self, data=None, **kwargs) -> None:
        """
        Create a camera module either through:
            option1 : Module(file)
            option2 : Module(freq=, F_lambda=)
            option3 : Module(wavelength=, F_lambda=)

        Parameters
        --------------------
        data : str or dict (option1)
            file path to csv file
            dict containing data
            has columns: x, y, pol (all in degrees)
        
        **kwargs
        Other keyword arguments as applicable to user-defined Module object 

        freq : frequency-like (option2)
            center of frequency band, default unit is GHz
            if each wafer is different (such as EoRSpec), pass as an iterable like [freq1, freq2, freq3]
        wavelength : distance-like (option2)
            intended wavelength of light, default unit is micron
            if each wafer is different (such as EoRSpec), pass as an iterable like [wavelength1, wavelength2, wavelength3]
        F_lambda : float (option2, option3) 
            factor for spacing between individual detectors, default is 1.4
        """

        # -- OPTION 1 --
        if not data is None:

            # using an existing Module object 
            if isinstance(data, str):
                self.data = pd.read_csv(data, index_col=False)
            # passing a dictionary (likely from Instrument class)
            else:
                self.data = pd.DataFrame(data)

            if not ('x' in self.data.columns or 'y' in self.data.columns):
                    raise ValueError(f'{data} must have x and y columns')

            if not 'pol' in self.data.columns:
                warnings.warn(f'{data} does not have a polarization column titled "pol"')

        # -- OPTION 2 --
        else:
            F_lambda = kwargs.pop('F_lambda', 1.4)

            if 'freq' in kwargs.keys():
                freq = u.Quantity(kwargs.pop('freq'), u.Hz*10**9).value
            elif 'wavelength' in kwargs.keys():
                wavelength = u.Quantity(kwargs.pop('wavelength'), u.micron).to(u.m).value
                freq = speed_of_light/wavelength/10**9
            else:
                raise ValueError('cannot create Module without one of file, freq, or wavelength')
            
            if kwargs:
                raise ValueError(f'unneccary keywords passed: {kwargs}')

            self.prop = {'freq': freq, 'F_lambda': F_lambda}
            self.data = self.generate_module(freq, F_lambda)        

    def to_csv(self, file, columns='all'):
        """
        Write Module object to file. If you want just the header, 

        Parameters
        ----------------------
        file : str
            file path to save location 
        columns : sequence or 'all'
            columns to save data to, default is 'all' columns (incl. rhombus and wafer number)
            otherwise, pass a list of columns e.g. ['x', 'y', 'pol']
        """

        if columns == 'all':
            columns = ['x', 'y', 'pol', 'rhombus', 'wafer']
        self.data.to_csv(file, columns=columns, index=False)

    def _waferpixelpos(p, numrows, numcols, numrhombus, centeroffset):
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

    @classmethod
    def generate_module(cls, center_freq, F_lambda=1.4):
        """
        Generates the pixel positions of a module centered at a particular frequency, based 
        off of code from the eor_spec_models package. The numerical values in the code are
        based off of the parameters for a detector array centered at 280 GHz. 

        Parameters
        ----------------------------------
        F_lambda : float 
            factor for spacing between individual detectors, default is 1.5
        center1, center2, center3 : GHz
            center of frequency band for each respective wafer, default is 280 GHz
        """

        # --- SETUP ---

        if isinstance(center_freq, int) or isinstance(center_freq, float):
            center1 = center2 = center3 = center_freq
        elif len(center_freq) == 3:
            center1 = center_freq[0]
            center2 = center_freq[1]
            center3 = center_freq[2]
        else:
            raise ValueError('center_freq is invalid')

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
        pixelcoords1 = cls._waferpixelpos(p1,numrows1,numcols1,numrhombus1,pixeloffset1)
        pixelcoords2 = cls._waferpixelpos(p2,numrows2,numcols2,numrhombus2,pixeloffset2)
        pixelcoords3 = cls._waferpixelpos(p3,numrows3,numcols3,numrhombus3,pixeloffset3)
        
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
        # λ/D in rad (D = 6m, diameter of the telescope)
        ang_res1 = F_lambda*math.degrees((3*10**8)/(center1*10**9)/6)
        ang_res2 = F_lambda*math.degrees((3*10**8)/(center2*10**9)/6)
        ang_res3 = F_lambda*math.degrees((3*10**8)/(center3*10**9)/6)

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
        return pd.DataFrame(
            np.append(np.append(pixelcoords1, pixelcoords2, axis=0), pixelcoords3, axis=0), 
            columns=['x', 'y', 'pol', 'rhombus', 'wafer']
        ).astype({'pol': np.int16, 'rhombus': np.uint8, 'wafer': np.uint8})

    # ATTRIBUTES 

    @property
    def x(self):
        return self.data['x'].to_numpy()*u.deg
    
    @property
    def y(self):
        return self.data['y'].to_numpy()*u.deg
    
    @property
    def pol(self):
        return self.data['pol'].to_numpy()*u.deg

# some standard modules FIXME will probably change in the future
CMBPol = Module(freq=350, F_lambda=1.4)
SFH = Module(freq=860, F_lambda=1.4)
EoRSpec = Module(freq=[262.5, 262.5, 367.5], F_lambda=1.4)
Mod280 = Module(freq=280, F_lambda=1.4)

#######################
#     INSTRUMENT
#######################

class Instrument():
    """
    Attributes
    ------------------------
    instr_offset : (deg, deg)
        offset in azimuth/elevation of the instrument from the boresight
    instr_rot : deg
        CCW rotation of the instrument
    modules : dict
        holds all the modules that populate the instrument with dict shape
        {identifier: {'module': Module, 'dist': deg, 'theta': deg, 'mod_rot': deg}, }
    slots : dict
        all default optics tube slots like {'i1': (dist, theta), }

    """

    def __init__(self, config_file=None, instr_offset=(0, 0), instr_rot=0) -> None:

        # config file is passed
        if not config_file is None:
            with open(config_file, 'r') as f:
                config = json.load(f)

            self._instr_offset = config['instr_offset']
            self._instr_rot = config['instr_rot']

            # populate modules
            self._modules = dict()
            for identifier in config['modules'].keys():
                self._modules[identifier] = {prop: config['modules'][identifier].pop(prop) for prop in ('dist', 'theta', 'mod_rot')}
                self._modules[identifier]['module'] = Module(config['modules'][identifier])

        # empty instrument
        else:
            self._instr_offset = u.Quantity(instr_offset, u.deg).value
            self._instr_rot = u.Quantity(instr_rot, u.deg).value
            
            # initialize empty dictionary of module objects 
            self._modules = dict()
    
    def save_config(self, path=None):
        """
        Saves as a dictionary like 
        {   'instr_offset': , 
            'instr_rot': , 
            'modules':  {
                module_name: {'dist':, 'theta':, 'mod_rot':, 'x': ,'y':, 'pol':},
            }
        }

        Parameters
        -------------
        path_or_buf : str or file handle, default None
            File path or object, if None is provided the result is returned as a dict.
        """

        # organize the configuration 
        config = {'instr_offset': list(self.instr_offset.value), 'instr_rot': self.instr_rot.value, 'modules': dict()}

        for identifier in self._modules.keys():
            config['modules'][identifier] = {
                'dist': self._modules[identifier]['dist'], 
                'theta': self._modules[identifier]['theta'],
                'mod_rot': self._modules[identifier]['mod_rot'],
                'x': list(self._modules[identifier]['module'].x.value),
                'y': list(self._modules[identifier]['module'].y.value),
                'pol': list(self._modules[identifier]['module'].pol.value)
            }
        
        # push configuration 
        if path is None:
            return config
        else:
            with open(path, 'w') as f:
                json.dump(config, f)

    def __repr__(self) -> str:
        instr_repr = f'instrument: offset {self.instr_offset}, rotation {self.instr_rot}\n------------------------------------'

        if len(self._modules) == 0:
            instr_repr += '\nempty'
        else:
            for identifier in self._modules.keys():
                instr_repr +=  "\n" + f"{identifier} \n (r, theta) = {(self._modules[identifier]['dist'], self._modules[identifier]['theta'])}, rotation = {self._modules[identifier]['mod_rot']}"
        
        return instr_repr

    def _check_identifier(func):
        def wrapper(self, identifier, *args, **kwargs):
            if not identifier in self._modules.keys():
                raise ValueError(f'identifier {identifier} is not valid')
            return func(self, identifier, *args, **kwargs)
        return wrapper

    # CHANGING INSTRUMENT CONFIGURATION

    def add_module(self, module, location, mod_rot=0, identifier=None):
        """
        Add a module to PrimeCam.

        Parameters
        -------------------------
        module : Module or str
            A Module object or one of the default options ['CMBPol', 'SFH', 'EoRSpec', 'Mod280']
        location : (distance [deg], theta [deg]) or str
            A tuple containing the module location from the center in polar coordinates (deg)
            or one of the default options ['c', 'i1', 'i2', 'o1', 'o2', etc] (see Ebina 2021 for picture)
        mod_rot : deg
            CCW rotation of the module, default is 0 deg
        idenitifier : str
            Name of the module. If user chose a default module option, then this this identifier
            will be its corresponging name unless otherwise specified. 
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
            location = self.slots[location]
        else:
            location = u.Quantity(location, u.deg).value

        # change module rotation
        mod_rot = u.Quantity(mod_rot, u.deg).value
        
        # if identifier is already a module that's saved
        if identifier in self._modules.keys():
            warnings.warn(f'Module {identifier} already exists. Overwriting it...')

        self._modules[identifier] = {'module': module, 'dist': location[0], 'theta': location[1], 'mod_rot': mod_rot}

    @_check_identifier
    def change_module(self, identifier, new_location=None, new_mod_rot=None, new_identifier=None, ):
    
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
                new_location = self.slots[new_location]
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
    def delete_module(self, identifier):
        self._modules.pop(identifier)

    # GETTERS 
    
    @_check_identifier
    def get_module(self, identifier):
        return self._modules[identifier]['module']

    @_check_identifier
    def get_dist(self, identitifer):
        return self._modules[identitifer]['dist']*u.deg
    
    @_check_identifier
    def get_theta(self, identifier):
        return self._modules[identifier]['theta']*u.deg
    
    @_check_identifier
    def get_mod_rot(self, identifier):
        return self._modules[identifier]['mod_rot']*u.deg

    def get_location(self, identifier):
        return (self.get_dist(identifier), self.get_theta(identifier))

    @property
    def instr_offset(self):
        return self._instr_offset*u.deg
    
    @property
    def instr_rot(self):
        return self._instr_rot*u.deg
    
    # SETTERS

    @instr_offset.setter
    def instr_offset(self, value):
        self._instr_offset = u.Quantity(value, u.deg).value

    @instr_rot.setter
    def instr_rot(self, value):
        self.instr_rot = u.Quantity(value, u.deg).value


class ModCam(Instrument): 
    slots = {'c': (0, 0)}

class PrimeCam(Instrument):

    # default configuration of optics tubes at 0 deg elevation 
    # in terms of (radius from center [deg], angle [deg])
    _default_ir = 1.78
    slots = {
        'c': (0, 0), 
        'i1': (_default_ir, -90), 'i2': (_default_ir, -30), 'i3': (_default_ir, 30), 'i4': (_default_ir, 90), 'i5': (_default_ir, 150), 'i6': (_default_ir, -150),
    }
    