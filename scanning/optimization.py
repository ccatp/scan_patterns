import math
from math import cos, sin

import numpy as np
import pandas as pd
import astropy.units as u
from fast_histogram import histogram2d

from scanning.camera import Module
from scanning.coordinates import TelescopePattern


class Simulation():
    """
    Attributes
    ------------
    _module
    _sky_pattern
    pxan
    param 
    max_pixel : number of pixels from center to one edge
    
    """

    def __init__(self, telescope_pattern, module, **kwargs) -> None:
        """
        Run a simulation given a TelescopePattern. 

        Parameters
        ----------------------------
        telescope_pattern : TelescopePattern
            A TelescopePattern object.
        module : str; Module
            If str: Gets the Module and AZ/ALT coordinates from a string indicating a module name in the instrument e.g. 'SFH'. 
            Recommended if telescope_pattern holds intrument data. 
            If Module: Uses telescope_pattern's "boresight" coordinates and uses module for its pixel positions. 
            Recommended if telescope_pattern does not hold instrument data. 
        
        **kwargs
        pixel_size : angle-like, default 10 arcsec, default unit arcsec
            Length of a square pixel on the hitmap. 
        max_acc : acceleration-like or None, default unit deg/s^2, default None
            Above this value in RA/DEC, data is considered unreliable. If None, all points are considered. 
        min_speed : speed-like or None, default unit deg/s, default None
            Below this value in RA/DEC, data is considered unreliable. If None, all points are considered. 
        hitmap_size : angle-like, default None
            Length and width of the resulting hitmap. 
        pols_on : iterable or None, default None
            Which default polarizations to keep on. If None, include all polarzations. 
        rhombi_on : iterable or None, default None
            Which rhombi to keep on. If None, include all rhombi. 
        wafers_on : iterable or None, default None
            Which wafers to keep on. If None, include all wafers. 
        pxan_list : iterable or None, default None
            A list of pixels such as [(x1, y1), (x2, y2)] to analyze. Each point (x, y) corresponds to a singular pixel that contains this point. 
            Measured as
            Cannot be used with pxan_lim. 
        pxan_lim : iterable or None, default None
            A recutangular range of pixels within: [(xmin, xmax), (ymin, ymax)].
            Cannot be used with pxan_list. 
        """

        # get correct TelescopePattern and Module
        if not isinstance(telescope_pattern, TelescopePattern):
            raise TypeError(f'telescope_pattern of type {type(telescope_pattern)} should be of type TelescopePattern')

        if isinstance(module, Module):
            self._module = pd.DataFrame(module.save_data())
        else:
            self._module = pd.DataFrame(telescope_pattern.instrument.get_module(module, with_rot=True).save_data())
            telescope_pattern = telescope_pattern.view_module(module)

        self._telescope_pattern = telescope_pattern
        self._sky_pattern = telescope_pattern.get_sky_pattern()

        # clean parameters
        self.param = self._clean_param(**kwargs)

        # preprocessing for hitmap
        self._preprocessing()
        rot_angle = self._telescope_pattern.rot_angle.to(u.rad).value

        # kept hits
        x_coord = self._sky_pattern.x_coord.value[self.validity_mask]/self.pixel_size
        y_coord = self._sky_pattern.y_coord.value[self.validity_mask]/self.pixel_size
        rot_angle_rad = rot_angle[self.validity_mask]
        print('Running simulation for kept hits...')
        self.sky_hist, self.det_hist, self.time_hist = self._simulate_scan(x_coord, y_coord, rot_angle_rad)

        # remvoed hits
        print('Running simulation for removed hits...')
        x_coord = self._sky_pattern.x_coord.value[~self.validity_mask]/self.pixel_size
        y_coord = self._sky_pattern.y_coord.value[~self.validity_mask]/self.pixel_size
        rot_angle_rad = rot_angle[~self.validity_mask]
        self.sky_hist_rem, self.det_hist_rem, self.time_hist_rem = self._simulate_scan(x_coord, y_coord, rot_angle_rad)

    def _clean_param(self, **kwargs):

        # regular parameters
        kwargs['pixel_size'] = u.Quantity(kwargs.get('pixel_size', 10), u.arcsec).to(u.deg).value
        kwargs['max_acc'] = u.Quantity(kwargs.get('max_acc', np.inf), u.deg/u.s/u.s).value
        kwargs['min_speed'] = u.Quantity(kwargs.get('min_speed', 0), u.deg/u.s).value
        kwargs['hitmap_size'] = u.Quantity(kwargs.get('hitmap_size', math.nan), u.arcsec).to(u.deg).value

        # parameters for keeping on certain parts of the module
        kwarg_keys = kwargs.keys()

        if 'pols_on' in kwarg_keys:
            all_pols = np.unique(self._module.pol)
            pols_on = kwargs['pols_on']
            if not set(pols_on).issubset(all_pols):
                raise ValueError(f'pols_on = {pols_on} is not a subset of available polarizations {all_pols}')
            kwargs['pols_on'] = np.array(pols_on)
        else:
            kwargs['pols_on'] = None
        
        if 'rhombi_on' in kwarg_keys:
            all_rhombi = np.unique(self._module.rhombi)
            rhombi_on = kwargs['rhombi_on']
            if not set(rhombi_on).issubset(all_rhombi):
                raise ValueError(f'rhombi_on = {rhombi_on} is not a subset of available polarizations {all_rhombi}')
            kwargs['rhombi_on'] = np.array(rhombi_on)
        else:
            kwargs['rhombi_on'] = None

        if 'wafers_on' in kwarg_keys:
            all_wafers = np.unique(self._module.wafer)
            wafers_on = kwargs['wafers_on']
            if not set(wafers_on).issubset(all_wafers):
                raise ValueError(f'wafers_on = {wafers_on} is not a subset of available polarizations {all_wafers}')
            kwargs['wafers_on'] = np.array(wafers_on)
        else:
            kwargs['wafers_on'] = None

        # pixel analysis

        if 'pxan_list' in kwarg_keys and 'pxan_lim' in kwarg_keys:
            raise TypeError('pxan_list and pxan_lim cannot be used at the same time')
        else: 
            self.pxan = 'pxan_list' in kwarg_keys or 'pxan_lim' in kwarg_keys

        if self.pxan:
            kwargs['pxan_norm'] = kwargs.get('pxan_norm', True)
            kwargs['pxan_list'] = kwargs.get('pxan_list')
            kwargs['pxan_lim'] = kwargs.get('pxan_lim')
        else:
            assert(not 'pxan_norm' in kwarg_keys)
    
        return kwargs

    def _preprocessing(self):

        # filtering out camera module elements

        mask = True

        if not self.pols_on is None:
            mask = mask & np.in1d(self._module.pol.value, self.pols_on)
        if not self.rhombi_on is None:
            mask = mask & np.in1d(self._module.rhombus, self.rhombi_on)
        if not self.wafers_on is None:
            mask = mask & np.in1d(self._module.wafer, self.wafers_on)

        if not mask is True:
            self._module = self._module[mask].reset_index(drop=True)

        # filtering out high accelerations and low velocities

        acc = self._sky_pattern.acc.value
        vel = self._sky_pattern.vel.value
        validity_mask = (acc < self.max_acc) & (vel > self.min_speed) 

        # get params for bin edges
        if math.isnan(self.hitmap_size): 
            farthest_det_elem = math.sqrt(max( (self._module.x.to_numpy())**2 + (self._module.y.to_numpy())**2 ))
            farthest_ts = math.sqrt(max( (self._sky_pattern.x_coord.value)**2 + (self._sky_pattern.y_coord.value)**2 ))
            self.max_pixel = math.ceil( (farthest_det_elem + farthest_ts)/self.pixel_size )
        else:
            self.max_pixel = math.ceil(self.hitmap_size/2/self.pixel_size)

        # pixel analysis
        if self.pxan:

            # pxan_lim
            if not self.pxan_lim is None:
                x_min = u.Quantity(self.pxan_lim[0][0], u.arcsec).to(u.deg).value
                x_min = math.floor(x_min/self.pixel_size)
                x_max = u.Quantity(self.pxan_lim[0][1], u.arcsec).to(u.deg).value
                x_max = math.ceil(x_max/self.pixel_size)

                y_min = u.Quantity(self.pxan_lim[1][0], u.arcsec).to(u.deg).value
                y_min = math.floor(y_min/self.pixel_size)
                y_max = u.Quantity(self.pxan_lim[1][1], u.arcsec).to(u.deg).value
                y_max = math.ceil(y_max/self.pixel_size)
                
                x_range_pxan = [ (x_min, x_max) ]
                y_range_pxan = [ (y_min, y_max) ]
                num_pxan = (x_max-x_min)*(y_max-y_min)

            # pxan_list
            else:
                x_range_pxan = []
                y_range_pxan = []
                for px in self.pxan_list:
                    x_min = u.Quantity(px[0], u.arcsec).to(u.deg).value
                    x_min = math.floor(x_min/self.pixel_size)
                    x_range_pxan.append( (x_min, x_min + 1) )

                    y_min = u.Quantity(px[1], u.arcsec).to(u.deg).value
                    y_min = math.floor(y_min/self.pixel_size)
                    y_range_pxan.append( (y_min, y_min + 1) )
                
                num_pxan = len(x_range_pxan)
        
        else:
            x_range_pxan = None
            y_range_pxan = None
            num_pxan = None

        # turn x and y in modules in terms of pixel location
        self._module['x'] = self._module['x']/self.pixel_size
        self._module['y'] = self._module['y']/self.pixel_size

        # save info
        self.validity_mask = validity_mask
        self.num_pxan = num_pxan
        self.x_range_pxan = x_range_pxan
        self.y_range_pxan = y_range_pxan

    def _simulate_scan(self, x_coord, y_coord, rot_angle_rad):

        num_bins = 2*self.max_pixel

        # get detector elements
        x_mod = self._module['x'].to_numpy()
        y_mod = self._module['y'].to_numpy()
        num_det_elem = len(x_mod)

        # number of timestamps
        num_ts = len(x_coord)

        # initialize histograms to be returned
        sky_hist = np.zeros((num_bins, num_bins))
        if self.pxan:
            det_hist = np.zeros(num_det_elem)
            time_hist = np.zeros(num_ts)
        else:
            det_hist = None
            time_hist = None

        # this section if for removed hits (if there are none)
        if num_ts == 0:
            return sky_hist, det_hist, time_hist

        # Divide process into chunks to abide by memory limits

        MEM_LIMIT = 8*10**7 
        chunk_ts = math.floor(MEM_LIMIT/num_det_elem)
        for chunk in range(math.ceil(num_ts/chunk_ts)):

            # initialize empty arrays (rows of ts and cols of det elements) to store hits 
            if (chunk+1)*chunk_ts <= num_ts:
                num_rows = chunk_ts
            else:
                num_rows = num_ts - chunk*chunk_ts

            all_x_coords = np.empty((num_rows, num_det_elem))
            all_y_coords = np.empty((num_rows, num_det_elem))

            # range of ts to loop over
            start = chunk*chunk_ts
            end = start + num_rows
            print(f'...{start}/{num_ts} completed...')

            # add all hits from the detector elements at each ts 
            for i, x_coord1, y_coord1, rot1 in zip(range(num_rows), x_coord[start:end], y_coord[start:end], rot_angle_rad[start:end]):
                all_x_coords[i] = x_coord1 + x_mod*cos(rot1) - y_mod*sin(rot1)
                all_y_coords[i] = y_coord1 + x_mod*sin(rot1) + y_mod*cos(rot1)

            sky_hist += histogram2d(all_x_coords, all_y_coords, range=[[-self.max_pixel, self.max_pixel], [-self.max_pixel, self.max_pixel]], bins=[num_bins, num_bins])

            # apply pixel(s) analysis
            if self.pxan:
                mask = False
                for x_range, y_range in zip(self.x_range_pxan, self.y_range_pxan):
                    mask = mask | ( (all_x_coords >= x_range[0]) & (all_x_coords < x_range[1]) & (all_y_coords >= y_range[0]) & (all_y_coords < y_range[1]) )

                det_hist += np.count_nonzero(mask, axis=0)
                time_hist[start:end] = np.count_nonzero(mask, axis=1)

        return sky_hist, det_hist, time_hist


    def _stats(self):
        # generate metrics for optimization (if not already)
        pass

    def __getattr__(self, attr):

        if attr in self.param.keys():
            return self.param[attr]
        else:
            raise AttributeError(f'attribtue {attr} not found')
    
    @property
    def std_dev(self):
        pass

