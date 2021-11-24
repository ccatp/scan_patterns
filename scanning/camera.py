# imports
import math
from math import sin, cos
import numpy as np
import pandas as pd
import warnings

""" - handling units (storing and checking), checking parameters type/config, error handling
- if polarization is in alternate direction
- Module csv file is just pixel positions and does not include pol
"""

#####################
#   CAMERA MODULE
#####################

class Module():
    """
    Class for a single camera module (e.g. EoRSpec, CMBPol, SFH). Each module 
    consists of three wafers/detector arrays. Each wafter contains three rhombuses. 
    
    Attributes
    ------------------
    mod_rot : deg
        CCW rotation of the module
    data : pd.DataFrame
        x (deg), y (deg), pol (deg int), rhombus (int), wafer (int)
    center_freq : GHz 
            center of frequency band
            if each wafer is different (such as EoRSpec), it is an iterable like [center1, center2, center3]
    F_lambda : float 
        factor for spacing between individual detectors
    """

    def __init__(self, file=None, center_freq=None, F_lambda=1.4, mod_rot=0) -> None:
        """
        Create a camera module either through:
            option1 : a csv file and module rotation
            option2 : pass center frequency, F_lambda and module rotation

        Parameters
        --------------------
        file : str (option1)
            file path to csv file with columns: x, y, pol, rhombus, wafer
        center_freq : GHz (option2)
            center of frequency band, default is 280 GHz
            if each wafer is different (such as EoRSpec), pass as an iterable like [center1, center2, center3]
        F_lambda : float (option2) 
            factor for spacing between individual detectors, default is 1.5
        mod_rot : deg
            CCW rotation of the module, default is 0 deg
        """

        self._mod_rot = mod_rot
        self._center_freq = center_freq 
        self._F_lambda = F_lambda
        
        if not center_freq is None:
            self.data = self.generate_module(center_freq, F_lambda)
        elif not file is None:
            self.data = pd.read_csv(file, index_col=False)
        else:
            raise ValueError('Value required for file or center.')
        
    def to_csv(self, file, columns='all', with_mod_rot=True):

        if columns == 'all':
            columns = ['x', 'y', 'pol', 'rhombus', 'wafer']

        if with_mod_rot:
            data_temp = self.data.copy()
            data_temp['x'] = self.x
            data_temp['y'] = self.y
            data_temp.to_csv(file, columns=columns, index=False)
        else:
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

    @property
    def mod_rot(self):
        return self._mod_rot
    
    @mod_rot.setter
    def mod_rot(self, value):
        self._mod_rot = value

    @property
    def center_freq(self):
        return self._center_freq

    @property
    def F_lambda(self):
        return self._F_lambda
    
    @property
    def ang_res(self):
        return self.F_lambda*math.degrees((3*10**8)/(self.center_freq*10**9)/6)

    @property
    def pitch(self):
        ratio = 280/self.center_freq
        return 2.75*10**-3 * ratio

    @property
    def x(self):
        mod_rot_rad = math.radians(self.mod_rot)
        return self.data['x'].to_numpy()*cos(mod_rot_rad) - self.data['y'].to_numpy()*sin(mod_rot_rad)
    
    @property
    def y(self):
        mod_rot_rad = math.radians(self.mod_rot)
        return self.data['x'].to_numpy()*sin(mod_rot_rad) + self.data['y'].to_numpy()*cos(mod_rot_rad)
    
    @property
    def pol(self):
        return self.data['pol'].to_numpy()

CMBPol = Module(center_freq=350)
SFH = Module(center_freq=860)
EoRSpec = Module(center_freq=[262.5, 262.5, 367.5])
Mod280 = Module(center_freq=280)

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
        {'module': Module, 'dist': deg, 'theta': deg}
    
    """

    def __init__(self, instr_offset=(0, 0), instr_rot=0) -> None:
        self._instr_offset = instr_offset
        self._instr_rot = instr_rot
        
        # initialize empty dictionary of module objects 
        self._modules = dict()
    
    def add_module(self, module, location, identifier=None, mod_rot=0):
        """
        Add a module to PrimeCam.

        Parameters
        -------------------------
        module : Module or str
            A Module object or one of the default options ['CMBPol', 'SFH', 'EoRSpec', 'Mod280']
        location : (distance [deg], theta [deg]) or str
            A tuple containing the module location from the center in polar coordinates (deg)
            or one of the default options ['c', 'i1', 'i2', 'o1', 'o2', etc] (see Ebina 2021 for picture)
        idenitifier : str
            Name of the module. If user chose a default module option, then this this identifier
            will be its corresponging name unless otherwise specified. 
        mod_rot : deg
            CCW rotation of the module, default is 0 deg
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
            location = self._default_config[location]
        
        # if identifier is already a module that's saved
        if identifier in self._modules.keys():
            warnings.warn(f'Module {identifier} already exists. Overwriting it...')

        # change module rotation
        module.mod_rot = mod_rot

        self._modules[identifier] = {'module': module, 'dist': location[0], 'theta': location[1]}

    def change_module(self, identifier, new_location=None, new_mod_rot=None, new_identifier=None, ):
        
        if not identifier in self._modules.keys():
            raise ValueError(f'identifier {identifier} is not valid')
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
                new_location = self._default_config[new_location]
            else:
                if new_location[0] is None:
                    new_location[0] = self._modules[identifier]['dist']
                if new_location[1] is None:
                    new_location[1] = self._modules[identifier]['theta']
            
            self._modules[identifier]['dist'] = new_location[0]
            self._modules[identifier]['theta'] = new_location[1]

        # change module rotation
        if not new_mod_rot is None:
            self._modules[identifier].mod_rot = new_mod_rot 

    def delete_module(self, identifier):
        if not identifier in self._modules.keys():
            raise ValueError(f'identifier {identifier} is not valid')

        self._modules.pop(identifier)

    def get_module(self, identifier):
        if not identifier in self._modules.keys():
            raise ValueError(f'identifier {identifier} is not valid')

        return self._modules[identifier]

    def get_module_location(self, identifier):
        if not identifier in self._modules.keys():
            raise ValueError(f'identifier {identifier} is not valid')

        return [self._modules[identifier]['dist'], self._modules[identifier]['theta']]

    def get_module_rotation(self, identifier):
        if not identifier in self._modules.keys():
            raise ValueError(f'identifier {identifier} is not valid')

        return self._modules[identifier]['module'].mod_rot

    def list_modules(self):
        
        for mod in self._modules.keys():
            print(f"""{mod} \n  (r, theta) = ({self._modules[mod]['dist']}, {self._modules[mod]['theta']}) | rotation = {self._modules[mod]['module'].mod_rot}""")

    @property
    def instr_offset(self):
        return self._instr_offset
    
    @instr_offset.setter
    def instr_offset(self, value):
        self._instr_offset = value
    
    @property
    def instr_rot(self):
        return self._instr_rot
    
    @instr_rot.setter
    def instr_rot(self, value):
        self.instr_rot = value

    @property
    def default_config(self):
        return self._default_config

class ModCam(Instrument): 
    _default_config = {'c': (0, 0)}

class PrimeCam(Instrument):

    # default configuration of optics tubes at 0 deg elevation 
    # in terms of (radius from center [deg], angle [deg])
    _default_ir = 1.78
    _default_or = 3.08
    _default_config = {
        'c': (0, 0), 
        'i1': (_default_ir, -90), 'i2': (_default_ir, -30), 'i3': (_default_ir, 30), 'i4': (_default_ir, 90), 'i5': (_default_ir, 150), 'i6': (_default_ir, -150),
        'o1': (_default_or, -60), 'o2': (_default_or), 'o3': (_default_or, 60), 'o4': (_default_or, 120), 'o5': (_default_or, 180), 'o6': (_default_or, -120) 
    }
    
