import math
from math import cos, sin
import csv
import json
from functools import wraps
from astropy.convolution.convolve import convolve_fft
from astropy.convolution.kernels import Gaussian2DKernel
from astropy.utils.misc import isiterable

import numpy as np
import pandas as pd
import astropy.units as u
from fast_histogram import histogram2d

import matplotlib.pyplot as plt

class Simulation():

    # other attributes (for development)
    # sky_hist, sky_hist_rem
    # det_hist, det_hist_rem
    # time_hist, time_hist_rem
    # pol_hist, pol_hist_rem

    # _sky_pattern
    # _field_rotation (degrees)
    # _module
    # _mem_limit

    # _pxan
    # _param
    # _prep
    # _det_mask
    # _validity_mask

    @property
    def pxan(self):
        return self._pxan
    
    @property
    def param(self):
        return self._param

    def __init__(self, telescope_pattern, module, sim_param=None, mem_limit=8*10**7, **kwargs) -> None:
        """
        Run a simulation on `telescope_pattern` with `module`. Use **kwargs to specify conditions. 
        Note that if multiple detector filters are used (`pols_on`, `det_lim`, etc.), all must be 
        satisifed. 

        Parameters
        ----------------------------
        telescope_pattern : TelescopePattern
            A TelescopePattern object.
        module : str or Module
            If `str`: Gets the Module and AZ/ALT coordinates from a string indicating a module name in the instrument e.g. 'SFH'. 
            Recommended if `telescope_pattern` holds intrument data. 
            If `Module`: Uses `telescope_pattern`'s "boresight" coordinates and uses `module` for its pixel positions. 
            Recommended if `telescope_pattern` does not hold instrument data. 
        mem_limit : 

        Keyword Args
        --------------------------
        pixel_size : float/Quantity/str; default 10 arcsec, default unit arcsec
            Length of a square pixel in x-y offsets. 
        max_acc : float/Quantity/str; default None, default unit deg/s^2
            Maximum absolute accleration in x-y offsets. If `None`, all points are considered valid.
        min_acc : float/Quantity/str; default None, default unit deg/s
            Minimum speed in x-y offsets. If `None`, all points are considered valid.
        
        pols_on : int sequence or None; default None
            Which polarization geometries (in degrees) of the detector elements to keep on. If `None`, include all. 
        rhombi_on : int sequence or None; default None
            Which rhombi to keep on. If `None`, include all rhombi. 
        wafers_on : int sequence or None; default None
            Which wafers to keep on. If `None`, include all wafers. 
        det_radius : two-length tuple or None; default None; default unit arcsec
            Take the (min_radius, max_radius) of detector elements to keep. Use `None` in the tuple for an unspecified limit. If `None`, include all. 
        det_lim : two-length sequence of two-length tuples or None; default None; default unit arcsec
            Take the [(x_min, x_max), (y_min, y_max)] range of dtector elements to keep. Use `None` in the tuple for an unspecified limit. If `None`, include all. 
        det_list : sequence of int, sequence of two-length tuples, or None; default None 
            If a sequence of `int`, a list of detector numbers to keep on exact detector elements.
            If a sequence of two-length tuples, a list of (x, y) positions to keep on the detector element closest to each position; default unit arcsec
            If `None, include all. 
        
        pxan_lim : two-length sequence of two-length tuples or None; default None; default unit arcsec
            A recutangular range of x-y pixels within [(xmin, xmax), (ymin, ymax)] to perform analysis.
            Cannot be used with `pxan_list`. If both are `None`, no pixel analysis is performed.
        pxan_list : sequence of two-length tuples, or None; default None; default unit arcsec
            A list of pixels such as [(x1, y1), (x2, y2)] to analyze. Each point (x, y) corresponds to a singular pixel that contains this point. 
            Cannot be used with `pxan_lim`. If both are `None`, no pixel analysis is performed.
        
        """

        self._mem_limit = mem_limit
        
        if isinstance(module, str):
            module_name = module
            module = telescope_pattern.instrument.get_module(module, with_instr_rot=True, with_mod_rot=True)
            telescope_pattern = telescope_pattern.view_module(module_name)
        
        # store sky pattern, field rotation, and module
        self._sky_pattern = telescope_pattern.get_sky_pattern()
        self._field_rotation = telescope_pattern.rot_angle.value
        self._module = module 

        # clean parameters
        if not sim_param is None:
            assert(not kwargs)
            with open(sim_param, 'r') as f:
                kwargs = json.load(f)

        self._param = self._clean_param(**kwargs)
        self._prep = self._preprocessing()

        # filer detector elements
        self._det_mask = self._filter_detectors()
        self._validity_mask = self._filter_ts()

        # initialize histograms
        self._sky_hist = self._sky_hist_rem = self._det_hist = self._det_hist_rem = self._time_hist = self._time_hist_rem = self._pol_hist = self._pol_hist_rem = None

    def _clean_param(self, **kwargs):
        new_kwargs = dict()

        new_kwargs['pixel_size'] = u.Quantity(kwargs.pop('pixel_size', 10), u.arcsec).to(u.deg).value

        max_acc = kwargs.pop('max_acc', None)
        new_kwargs['max_acc'] = u.Quantity(max_acc, u.deg/u.s/u.s).value if not max_acc is None else None

        min_speed = kwargs.pop('min_speed', None)
        new_kwargs['min_speed'] = u.Quantity(min_speed, u.deg/u.s).value if not min_speed is None else None

        # parameters for keeping on certain parts of the module

        pols_on = kwargs.pop('pols_on', None)
        if not pols_on is None:
            all_pols = np.unique(self._module.pol.value)
            if not set(pols_on).issubset(all_pols):
                raise ValueError(f'pols_on = {pols_on} is not a subset of available polarizations {all_pols}')
        new_kwargs['pols_on'] = pols_on
  
        rhombi_on = kwargs.pop('rhombi_on', None)
        if not rhombi_on is None:
            all_rhombi = np.unique(self._module.rhombus)
            if not set(rhombi_on).issubset(all_rhombi):
                raise ValueError(f'rhombi_on = {rhombi_on} is not a subset of available rhombuses {all_rhombi}')
        new_kwargs['rhombi_on'] = rhombi_on

        wafers_on = kwargs.pop('wafers_on', None)
        if not wafers_on is None:
            all_wafers = np.unique(self._module.wafer)
            if not set(wafers_on).issubset(all_wafers):
                raise ValueError(f'wafers_on = {wafers_on} is not a subset of available wafers {all_wafers}')
        new_kwargs['wafers_on'] = wafers_on
        
        det_radius = kwargs.pop('det_radius', None)
        if not det_radius is None:
            min_radius = det_radius[0]
            min_radius = -math.inf if min_radius is None else u.Quantity(min_radius, u.arcsec).to(u.deg).value
            max_radius = det_radius[1]
            max_radius = math.inf if max_radius is None else u.Quantity(max_radius, u.arcsec).to(u.deg).value
            new_kwargs['det_radius'] = (min_radius, max_radius)
        else:
            new_kwargs['det_radius'] = None
        
        det_lim = kwargs.pop('det_lim', None)
        if not det_lim is None:
            min_x = det_lim[0][0]
            min_x = -math.inf if min_x is None else u.Quantity(min_x, u.arcsec).to(u.deg).value
            max_x = det_lim[0][1]
            max_x = math.inf if max_x is None else u.Quantity(max_x, u.arcsec).to(u.deg).value
            min_y = det_lim[1][0]
            min_y = -math.inf if min_y is None else u.Quantity(min_y, u.arcsec).to(u.deg).value
            max_y = det_lim[1][1]
            max_y = math.inf if max_y is None else u.Quantity(max_y, u.arcsec).to(u.deg).value
            new_kwargs['det_lim'] = ((min_x, max_x), (min_y, max_y))
        else:
            new_kwargs['det_lim'] = None
        
        det_list = kwargs.pop('det_list', None)
        if not det_list is None:
            try:
                new_det_list = []
                for point in det_list:
                    new_det_list.append( (u.Quantity(point[0], u.arcsec).to(u.deg).value, u.Quantity(point[1], u.arcsec).to(u.deg).value) )
                new_kwargs['det_list'] = np.array(new_det_list)
            except TypeError:
                assert(len(np.shape(det_list)) == 1)
                new_kwargs['det_list'] = np.array(det_list)

        else:
            new_kwargs['det_list'] = None

        # for pixel analysis

        self._pxan = False
        pxan_list = kwargs.pop('pxan_list', None)
        pxan_lim = kwargs.pop('pxan_lim', None)

        if not pxan_list is None and not pxan_lim is None:
            raise TypeError('pxan_list and pxan_lim cannot be used at the same time')

        elif not pxan_lim is None:
            self._pxan = True
            x_min = u.Quantity(pxan_lim[0][0], u.arcsec).to(u.deg).value
            x_max = u.Quantity(pxan_lim[0][1], u.arcsec).to(u.deg).value
            y_min = u.Quantity(pxan_lim[1][0], u.arcsec).to(u.deg).value
            y_max = u.Quantity(pxan_lim[1][1], u.arcsec).to(u.deg).value
            new_kwargs['pxan_lim']= [(x_min, x_max), (y_min, y_max)]

        elif not pxan_list is None:
            self._pxan = True
            new_pxan_list = []
            for px in pxan_list:
                x = u.Quantity(px[0], u.arcsec).to(u.deg).value
                y = u.Quantity(px[1], u.arcsec).to(u.deg).value
                new_pxan_list.append( (x, y) )
            new_kwargs['pxan_list'] = new_pxan_list

        # return
        if kwargs:
            raise TypeError(f'uncessary keywords: {kwargs.keys()}')

        return new_kwargs

    def _preprocessing(self):
        pixel_size = self._param['pixel_size']

        farthest_det_elem = math.sqrt(max( (self._module.x.value)**2 + (self._module.y.value)**2 ))
        farthest_ts = math.sqrt(max(self._sky_pattern.distance.value))
        max_pixel = math.ceil((farthest_det_elem + farthest_ts)/pixel_size)

        if self._pxan:

            pxan_lim = self._param.get('pxan_lim')
            pxan_list = self._param.get('pxan_list')

            if not pxan_lim is None:
                x_min = math.floor(pxan_lim[0][0]/pixel_size)
                x_max = math.ceil(pxan_lim[0][1]/pixel_size)
                y_min = math.floor(pxan_lim[1][0]/pixel_size)
                y_max = math.ceil(pxan_lim[1][1]/pixel_size)
                x_range_pxan = [ (x_min, x_max) ]
                y_range_pxan = [ (y_min, y_max) ]
                num_pxan = (x_max-x_min)*(y_max-y_min)
            else:
                x_range_pxan = []
                y_range_pxan = []
                for px in pxan_list:
                    x_min = math.floor(px[0]/pixel_size)
                    x_range_pxan.append( (x_min, x_min + 1) )

                    y_min = math.floor(px[1]/pixel_size)
                    y_range_pxan.append( (y_min, y_min + 1) )
                
                num_pxan = len(x_range_pxan)
        else:
            x_range_pxan = y_range_pxan = num_pxan = None
        

        return {'max_pixel': max_pixel, 'x_range_pxan': x_range_pxan, 'y_range_pxan': y_range_pxan, 'num_pxan': num_pxan}

    def _filter_detectors(self):
        
        module = self._module
        det_mask = np.full(module.x.value.size, True)
        param = self._param

        if not param.get('pols_on') is None:
            det_mask = np.in1d(module.pol.value, param['pols_on'])
        if not param.get('rhombi_on') is None:
            det_mask = det_mask & np.in1d(module.rhombus, param['rhombi_on'])
        if not param.get('wafers_on') is None:
            det_mask = det_mask & np.in1d(module.wafer, param['wafers_on'])
        
        if not param.get('det_radius') is None:
            radius = np.sqrt(module.x.value**2 + module.y.value**2)
            det_mask = det_mask & (radius >= param['det_radius'][0]) & (radius <= param['det_radius'][1])
        if not param.get('det_lim') is None:
            det_lim = param['det_lim']
            det_mask = det_mask & (module.x.value >= det_lim[0][0]) & (module.x.value <= det_lim[0][1]) & (module.y.value >= det_lim[1][0]) & (module.y.value <= det_lim[1][1])
        if not param.get('det_list') is None:
            det_list = param['det_list']
            if len(np.shape(det_list)) == 1:
                det_mask = det_mask & np.in1d(module.pixel_num, det_list)
            else:
                det_list_mask = np.full(module.x.value.size, False)
                for point in det_list:
                    distance_away = np.sqrt((module.x.value - point[0])**2 + (module.y.value - point[1])**2)
                    point_i = distance_away.argmin()
                    det_list_mask[point_i] = True
                det_mask = det_mask & det_list_mask

        return det_mask

    def _filter_ts(self):
        param = self._param
        
        # filter ts (rows)
        ts_mask = np.full(self._sky_pattern.x_coord.value.size, True)
        if not param.get('max_acc') is None:
            ts_mask = abs(self._sky_pattern.acc.value) <= param['max_acc']
        if not param.get('min_speed') is None:
            ts_mask = abs(self._sky_pattern.vel.value) >= param['min_speed']

        return ts_mask
    
    def _set_histograms(self, kept):

        # select (in)valid points
        validity_mask = self._validity_mask
        if not kept:
            validity_mask = ~validity_mask

        # prep x_coord, y_coord, and rot_angle for simulating
        pixel_size = self._param['pixel_size']
        x_coord_px = self._sky_pattern.x_coord.value[validity_mask]/pixel_size
        y_coord_px = self._sky_pattern.y_coord.value[validity_mask]/pixel_size
        rot_angle_rad = np.radians(self._field_rotation[validity_mask])

        sky_hist, det_hist, time_hist = self._simulate_scan(x_coord_px, y_coord_px, rot_angle_rad)

        # clean up histograms
        hitmap_range = np.arange(-self._prep['max_pixel'], self._prep['max_pixel'])
        sky_hist = pd.DataFrame(sky_hist, index=hitmap_range, columns=hitmap_range)

        if self.pxan:
            det_num = self._module.pixel_num[self._det_mask]
            det_hist = pd.Series(det_hist, index=det_num).reindex(self._module.pixel_num)

            sky_pattern_index = np.arange(self._sky_pattern.x_coord.value.size)
            time_num = sky_pattern_index[validity_mask]
            time_hist = pd.Series(time_hist, index=time_num).reindex(sky_pattern_index)

        # store histograms
        if kept:
            self._sky_hist = sky_hist
            self._det_hist = det_hist
            self._time_hist = time_hist
        else:
            self._sky_hist_rem = sky_hist
            self._det_hist_rem = det_hist
            self._time_hist_rem = time_hist

    def _simulate_scan(self, x_coord_px, y_coord_px, rot_angle_rad):

        max_pixel = self._prep['max_pixel']
        num_bins = 2*max_pixel

        x_range_pxan = self._prep['x_range_pxan']
        y_range_pxan = self._prep['y_range_pxan']

        # get detector elements
        pixel_size = self._param['pixel_size']
        x_mod = self._module.x.value[self._det_mask]/pixel_size
        y_mod = self._module.y.value[self._det_mask]/pixel_size
        num_det_elem = len(x_mod)

        # number of timestamps
        num_ts = len(x_coord_px)

        # initialize histograms to be returned
        sky_hist = np.zeros((num_bins, num_bins))
        if self.pxan:
            det_hist = np.zeros(num_det_elem)
            time_hist = np.zeros(num_ts)
        else:
            det_hist = None
            time_hist = None

        # this section if for removed hits (if there are none)
        if num_ts == 0 or num_det_elem == 0:
            return sky_hist, det_hist, time_hist

        # Divide process into chunks to abide by memory limits

        chunk_ts = math.floor(self._mem_limit/num_det_elem)
        for chunk in range(math.ceil(num_ts/chunk_ts)):

            # initialize empty arrays (rows of ts and cols of det elements) to store hits 
            if (chunk+1)*chunk_ts <= num_ts:
                num_rows = chunk_ts
            else:
                num_rows = num_ts - chunk*chunk_ts

            all_x_coords = np.empty((num_rows, num_det_elem), dtype=np.float16)
            all_y_coords = np.empty((num_rows, num_det_elem), dtype=np.float16)

            # range of ts to loop over
            start = chunk*chunk_ts
            end = start + num_rows
            print(f'...{start}/{num_ts} completed...')

            # add all hits from the detector elements at each ts 
            for i, x_coord1, y_coord1, rot1 in zip(range(num_rows), x_coord_px[start:end], y_coord_px[start:end], rot_angle_rad[start:end]):
                all_x_coords[i] = x_coord1 + x_mod*cos(rot1) - y_mod*sin(rot1)
                all_y_coords[i] = y_coord1 + x_mod*sin(rot1) + y_mod*cos(rot1)

            sky_hist += histogram2d(all_x_coords, all_y_coords, range=[[-max_pixel, max_pixel], [-max_pixel, max_pixel]], bins=[num_bins, num_bins])

            # apply pixel(s) analysis
            if self.pxan:
                mask = False
                for x_range, y_range in zip(x_range_pxan, y_range_pxan):
                    mask = mask | ( (all_x_coords >= x_range[0]) & (all_x_coords < x_range[1]) & (all_y_coords >= y_range[0]) & (all_y_coords < y_range[1]) )

                det_hist += np.count_nonzero(mask, axis=0)
                time_hist[start:end] = np.count_nonzero(mask, axis=1)

        return sky_hist.T, det_hist, time_hist

    def _check_histograms(func):
        @wraps(func)
        def wrapper(self, hits, *args, **kwargs):
            if hits == 'total':
                if self._sky_hist is None:
                    self._set_histograms(True)
                if self._sky_hist_rem is None:
                    self._set_histograms(False)
            
            elif hits == 'kept':
                if self._sky_hist is None:
                    self._set_histograms(True)
            
            elif hits == 'removed':
                if self._sky_hist_rem is None:
                    self._set_histograms(False)
            return func(self, hits, *args, **kwargs)
        return wrapper

    @_check_histograms
    def sky_histogram(self, hits='kept', convolve=True, norm_time=False, path=None):
        if hits == 'kept':
            sky_hist = self._sky_hist.to_numpy()
        elif hits == 'removed':
            sky_hist = self._sky_hist_rem.to_numpy()
        else:
            sky_hist = self._sky_hist.to_numpy() + self._sky_hist_rem.to_numpy()

        pixel_size = self._param['pixel_size']
        
        # convolution
        if convolve:
            ang_res = self._module.ang_res.value
            if isiterable(ang_res):
                raise ValueError('Unable to convolve since ang_res is not constant.')
            stddev = (ang_res/pixel_size)/np.sqrt(8*np.log(2))
            kernel = Gaussian2DKernel(stddev)
            sky_hist = convolve_fft(sky_hist, kernel, boundary='fill', fill_value=0)
    
        # normalize time
        if norm_time:
            total_time = self._sky_pattern.scan_duration.value
            sky_hist = sky_hist/total_time
        
        # return
        hitmap_range = np.arange(-self._prep['max_pixel'], self._prep['max_pixel'])*pixel_size

        if path is None:
            return sky_hist, hitmap_range
        else:
            sky_hist = pd.DataFrame(sky_hist, index=hitmap_range, columns=hitmap_range)
            sky_hist.to_csv(path, index=True, header=True)
                
    @_check_histograms
    def det_histogram(self, hits='kept', norm_pxan=True, norm_time=False, path=None):
        if hits == 'kept':
            det_hist = self._det_hist
        elif hits == 'removed':
            det_hist = self._det_hist_rem
        else:
            det_hist = self._det_hist + self._det_hist_rem
        
        # divide by number of pixels in pixel analysis
        if norm_pxan:
            num_pxan = self._prep['num_pxan']
            det_hist = det_hist/num_pxan
    
        # normalize time
        if norm_time:
            total_time = self._sky_pattern.scan_duration.value
            det_hist = det_hist/total_time
        
        if path is None:
            return det_hist.values
        else:
            det_hist.to_csv(path, index=True)

    @_check_histograms
    def time_histogram(self, hits='kept', norm_pxan=True, norm_time=False, path=None):
        if hits == 'kept':
            time_hist = self._time_hist.values
        elif hits == 'removed':
            time_hist = self._time_hist_rem.values
        else:
            time_hist = np.nan_to_num(self._time_hist) + np.nan_to_num(self._time_hist_rem)
        
        # divide by number of pixels in pixel analysis
        if norm_pxan:
            num_pxan = self._prep['num_pxan']
            time_hist = time_hist/num_pxan
    
        # normalize time
        if norm_time:
            total_time = self._sky_pattern.scan_duration.value
            time_hist = time_hist/total_time
        
        if path is None:
            return time_hist
        else:
            pd.Series(time_hist).to_csv(path, index=True)

    def pol_histogram(self, hits='kept', path=None):
        
        if hits == 'kept':
            field_rotation = self._field_rotation[self._validity_mask]
        elif hits =='removed':
            field_rotation = self._field_rotation[~self._validity_mask]
        else:
            field_rotation = self._field_rotation
        
        all_pol, all_pol_counts = np.unique(self._module.pol.value[self._det_mask], return_counts=True)

        pol_grid, rot_grid = np.meshgrid(all_pol, field_rotation)
        total_rot =  pd.DataFrame((pol_grid + rot_grid)%90, columns=all_pol) 

        if path is None:
            return total_rot, all_pol_counts
        else:
            total_rot.to_csv(path, index=True)

        