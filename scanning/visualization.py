from math import pi, sin, cos, tan, radians
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.utils import isiterable
import astropy.units as u
from astropy.convolution import Gaussian2DKernel, convolve_fft

from scanning.coordinates import SkyPattern, TelescopePattern
from scanning import FYST_LOC

# Modules and Instruments

def plot_module(mod, mode=None, path=None, show_plot=True):
    """
    Plot the pixel positions and default polarizations of a given module. 

    Parameters
    -----------------------------
    mod : Module
        A Module object. 
    mode : str or None, default None
        None for just the pixel positions.
        'pol' to include default orientations.
        'rhomubs' to include rhombus info.
        'wafer' to include wafer info. 
    path : str or None, default None
        If not None, saves the image to the file path. 
    show_plot : bool, default True
        Whether to display the resulting figure.
    """

    fig_mod = plt.figure('Module', figsize=(8, 8))
    ax_mod = plt.subplot2grid((1, 1), (0, 0))
    x_deg = mod.x.value
    y_deg = mod.y.value

    if mode is None:
        plot_mod = ax_mod.scatter(x_deg, y_deg, s=1)
    else:

        # mode
        if mode == 'pol':
            categories = np.unique(mod.pol.value)
            num_def = len(categories)
            y_label = 'Default Orientation [deg]'
            tick_locs = (categories + 7.5)*(num_def-1)/num_def
            c = mod.pol
        elif mode == 'rhombus':
            categories = np.unique(mod.rhombus)
            num_def = len(categories)
            y_label = 'Rhombus'
            tick_locs = (categories + 0.5)*(num_def-1)/num_def
            c = mod.rhombus
        elif mode == 'wafer':
            categories = np.unique(mod.wafer)
            num_def = len(categories)
            y_label = 'Wafer'
            tick_locs = (categories + 0.5)*(num_def-1)/num_def
            c = mod.wafer
        else:
            raise ValueError(f'mode={mode} is not valid')
        
        # determine which colormap to use
        if num_def <= 9:
            cmap = plt.cm.get_cmap('Set1', num_def)
        else:
            cmap = plt.cm.get_cmap('hsv', num_def)

        # plot
        plot_mod = ax_mod.scatter(x_deg, y_deg, c=c, cmap=cmap, s=1) # FIXME make pixel sizes proportional

        # color bar
        cbar = plt.colorbar(plot_mod, fraction=0.046, pad=0.04)
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(categories)
        cbar.ax.set_ylabel(y_label)

    ax_mod.set_aspect('equal', 'box')
    ax_mod.set(xlabel='x offset (deg)', ylabel='y offset (deg)', title='Map of Detector Array')
    ax_mod.grid()
    fig_mod.tight_layout()

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()

def instrument_config(instrument, path=None, show_plot=True):
    """
    Save a plot of the instrument configuration. 

    Parameters
    ------------------------
    instrument : Instrument
        An Instrument object. 
    path : str or None, default None
        If not None, saves the image to the file path. 
    show_plot : bool, default True
        Whether to display the resulting figure.
    """

    fig = plt.figure('Insturment', figsize=(8, 8))
    ax = plt.subplot2grid((1, 1), (0, 0))

    # instrument rotation and offset
    instr_offset = instrument.instr_offset.value
    instr_rot = radians(instrument.instr_rot.value)

    # default slots 
    n=100
    radius = 1.3/2
    for slot_name, slot_loc in instrument.slots.items():

        # get slots into correct position
        x_offset = slot_loc[0]*cos(radians(slot_loc[1]))
        y_offset = slot_loc[0]*sin(radians(slot_loc[1]))

        circle1_x = np.array([cos(2*pi/n*x)*radius for x in range(0,n+1)]) + x_offset 
        circle1_y = np.array([sin(2*pi/n*x)*radius for x in range(0,n+1)]) + y_offset

        # apply instr rot/offset
        circle2_x = circle1_x*cos(instr_rot) - circle1_y*sin(instr_rot) + instr_offset[0]
        circle2_y = circle1_x*sin(instr_rot) + circle1_y*cos(instr_rot) + instr_offset[1]

        ax.plot(circle2_x, circle2_y, ls='dashed', color='black')
        
    # modules 
    for identifier, value in instrument._modules.items():
        mod = instrument.get_module(identifier, with_rot=True)

        # get modules into correct position
        x_offset = value['dist']*cos(radians(value['theta']))
        y_offset = value['dist']*sin(radians(value['theta']))
        
        x1 = mod.x.value + instr_offset[0] + x_offset
        y1 = mod.y.value + instr_offset[1] + y_offset 
        
        plt.scatter(x1, y1, s=0.5, label=identifier)
        
    ax.grid()
    ax.legend()
    ax.set_aspect('equal')
    ax.set(xlabel='x offset from boresight [deg]', ylabel='y offset from boresight [deg]', title='Instrument Configuration')

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()

# Sky Patterns and Telescope Patterns

def sky_path(pattern, module=None, path=None, show_plot=True):

    fig_coord = plt.figure('coordinates', figsize=(8, 8))
    ax_coord = plt.subplot2grid((1, 1), (0, 0))

    # pattern is SkyPattern

    if isinstance(pattern, SkyPattern):
        ax_coord.plot(pattern.x_coord.value, pattern.y_coord.value)
        ax_coord.set(xlabel='x offset [deg]', ylabel='x offset [deg]', title=f'Sky Path')

    # pattern is TelescopePattern
    elif isinstance(pattern, TelescopePattern):
        ax_coord.set(xlabel='RA [deg]', ylabel='DEC [deg]', title=f'Sky Path')
        if module is None:
            module = ['boresight']
        elif not isiterable(module):
            module = [module]

        for m in module:
            if m == 'boresight' or m == (0, 0):
                temp = pattern
            else:
                temp = pattern.view_module(m)

            ax_coord.plot(temp.ra_coord.value, temp.dec_coord.value, label=m)
        
        ax_coord.legend()
    else:
        raise TypeError('pattern is not of correct type')
    
    ax_coord.set_aspect('equal', 'box')
    ax_coord.grid()

    fig_coord.tight_layout()

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()

def telescope_path(telescope_pattern, module=None, path=None, show_plot=True):

    if module is None:
        module = ['boresight']
    elif not isiterable(module):
        module = [module]
    
    fig_coord = plt.figure('coordinates', figsize=(8, 8))
    ax_coord = plt.subplot2grid((1, 1), (0, 0))

    for m in module:
        if m == 'boresight' or m == (0, 0):
            temp = telescope_pattern
        else:
            temp = telescope_pattern.view_module(m)

        ax_coord.plot(temp.az_coord.value, temp.alt_coord.value, label=m)

    ax_coord.set_aspect('equal', 'box')
    ax_coord.set(xlabel='Azimuth [deg]', ylabel='Elevation [deg]', title=f'Telescope Path')
    ax_coord.grid()
    ax_coord.legend()

    fig_coord.tight_layout()

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()
    
def telescope_kinematics(telescope_pattern, module=None, plots=['coord', 'vel', 'acc', 'jerk'], path=None, show_plot=True):

    if not module is None:
        telescope_pattern = telescope_pattern.view_module(module)

    num_plots = len(plots)
    i = 0 

    fig_motion = plt.figure('motion', figsize=(8, 2*num_plots))

    # time vs. azmiuth/elevation
    if 'coord' in plots:
        ax_coord = plt.subplot2grid((num_plots, 1), (i, 0))
        start_az = telescope_pattern.az_coord.value[0]
        start_alt = telescope_pattern.alt_coord.value[0]

        ax_coord.plot(telescope_pattern.time_offset.value, (telescope_pattern.az_coord.value - start_az)*cos(radians(start_alt)), label=f'AZ from {round(start_az, 2)} deg')
        ax_coord.plot(telescope_pattern.time_offset.value, telescope_pattern.alt_coord.value - start_alt, label=f'EL from {round(start_alt, 2)} deg')
        ax_coord.legend(loc='upper right')
        ax_coord.set(xlabel='Time Offset [s]', ylabel='Position [deg]', title=f'Time vs. Position')
        ax_coord.grid()
        i += 1

    # velocity
    if 'vel' in plots:
        ax_vel = plt.subplot2grid((num_plots, 1), (i, 0), sharex=ax_coord)
        ax_vel.plot(telescope_pattern.time_offset.value, telescope_pattern.vel.value, label='Total', c='black', ls='dashed', alpha=0.25)
        ax_vel.plot(telescope_pattern.time_offset.value, telescope_pattern.az_vel.value, label='AZ')
        ax_vel.plot(telescope_pattern.time_offset.value, telescope_pattern.alt_vel.value, label='EL')
        ax_vel.legend(loc='upper right')
        ax_vel.set(xlabel='Time offset [s]', ylabel='Velocity [deg/s]', title=f'Time vs. Velocity')
        ax_vel.grid()
        i += 1

    # acceleration
    if 'acc' in plots:
        ax_acc = plt.subplot2grid((num_plots, 1), (i, 0), sharex=ax_coord)
        ax_acc.plot(telescope_pattern.time_offset.value, telescope_pattern.acc.value, label='Total', c='black', ls='dashed', alpha=0.25)
        ax_acc.plot(telescope_pattern.time_offset.value, telescope_pattern.az_acc.value, label='AZ')
        ax_acc.plot(telescope_pattern.time_offset.value, telescope_pattern.alt_acc.value, label='EL')
        ax_acc.legend(loc='upper right')
        ax_acc.set(xlabel='Time offset [s]', ylabel='Acceleration [deg/s^2]', title=f'Time vs. Acceleration')
        ax_acc.grid()
        i += 1

    # jerk
    if 'jerk' in plots:
        ax_jerk = plt.subplot2grid((num_plots, 1), (i, 0), sharex=ax_coord)
        ax_jerk.plot(telescope_pattern.time_offset.value, telescope_pattern.jerk.value, label='Total', c='black', ls='dashed', alpha=0.25)
        ax_jerk.plot(telescope_pattern.time_offset.value, telescope_pattern.az_jerk.value, label='AZ')
        ax_jerk.plot(telescope_pattern.time_offset.value, telescope_pattern.alt_jerk.value, label='EL')
        ax_jerk.legend(loc='upper right')
        ax_jerk.set(xlabel='Time Offset (s)', ylabel='Jerk [deg/s^2]', title=f'Time vs. Jerk')
        ax_jerk.grid()
        i += 1

    fig_motion.tight_layout()

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()
    
def sky_kinematics(pattern, module=None, plots=['coord', 'vel', 'acc', 'jerk'], path=None, show_plot=True):

    if isinstance(pattern, TelescopePattern):
        if module is None:
            pattern = pattern.get_sky_pattern()
        else:
            pattern = pattern.view_module(module).get_sky_pattern()

    num_plots = len(plots)
    i = 0 

    fig_motion = plt.figure('motion', figsize=(8, 2*num_plots))

    # time vs. azmiuth/elevation
    if 'coord' in plots:
        ax_coord = plt.subplot2grid((num_plots, 1), (i, 0))
        ax_coord.plot(pattern.time_offset.value, pattern.x_coord.value, label=f'x offset [deg]')
        ax_coord.plot(pattern.time_offset.value, pattern.y_coord.value, label=f'y offset [deg]')
        ax_coord.legend(loc='upper right')
        ax_coord.set(xlabel='Time Offset [s]', ylabel='Position [deg]', title=f'Time vs. Position')
        ax_coord.grid()
        i += 1

    # velocity
    if 'vel' in plots:
        ax_vel = plt.subplot2grid((num_plots, 1), (i, 0), sharex=ax_coord)
        ax_vel.plot(pattern.time_offset.value, pattern.vel.value, label='Total', c='black', ls='dashed', alpha=0.25)
        ax_vel.plot(pattern.time_offset.value, pattern.x_vel.value, label='x')
        ax_vel.plot(pattern.time_offset.value, pattern.y_vel.value, label='y')
        ax_vel.legend(loc='upper right')
        ax_vel.set(xlabel='Time offset [s]', ylabel='Velocity [deg/s]', title=f'Time vs. Velocity')
        ax_vel.grid()
        i += 1

    # acceleration
    if 'acc' in plots:
        ax_acc = plt.subplot2grid((num_plots, 1), (i, 0), sharex=ax_coord)
        ax_acc.plot(pattern.time_offset.value, pattern.acc.value, label='Total', c='black', ls='dashed', alpha=0.25)
        ax_acc.plot(pattern.time_offset.value, pattern.x_acc.value, label='x')
        ax_acc.plot(pattern.time_offset.value, pattern.y_acc.value, label='y')
        ax_acc.legend(loc='upper right')
        ax_acc.set(xlabel='Time offset [s]', ylabel='Acceleration [deg/s^2]', title=f'Time vs. Acceleration')
        ax_acc.grid()
        i += 1

    # jerk
    if 'jerk' in plots:
        ax_jerk = plt.subplot2grid((num_plots, 1), (i, 0), sharex=ax_coord)
        ax_jerk.plot(pattern.time_offset.value, pattern.jerk.value, label='Total', c='black', ls='dashed', alpha=0.25)
        ax_jerk.plot(pattern.time_offset.value, pattern.x_jerk.value, label='x')
        ax_jerk.plot(pattern.time_offset.value, pattern.y_jerk.value, label='y')
        ax_jerk.legend(loc='upper right')
        ax_jerk.set(xlabel='Time Offset (s)', ylabel='Jerk [deg/s^2]', title=f'Time vs. Jerk')
        ax_jerk.grid()
        i += 1

    fig_motion.tight_layout()

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()

# Hitmaps and Simulation 

def hitmap(sim, convolve=True, norm_time=False, total_max=None, kept_max=None, rem_max=None, path=None, show_plot=True):
    """
    Parameters
    ---------------------------------------------
    sim : Simulation
        A simulation object. 
    convolve : bool, default True
        Whether to convolve the hitmap. 
    norm_time : bool, default False
        True for hits/px/sec. False for hits/px.
    total_max, kept_max, rem_max : float, default None
        Set maximums for the color bars. 
    path : str or None, default None
        If not None, saves the image to the file path. 
    show_plot : bool, default True
        Whether to display the resulting figure.
    """

    sky_hist = sim.sky_hist
    sky_hist_rem = sim.sky_hist_rem

    pixel_size = sim.pixel_size
    max_pixel = sim.max_pixel
    max_pixel_deg = max_pixel*pixel_size

    # convolution
    if convolve:
        beam_size = (50*u.arcsec).to(u.deg).value
        stddev = (beam_size/pixel_size)/np.sqrt(8*np.log(2))
        kernel = Gaussian2DKernel(stddev)
        sky_hist = convolve_fft(sky_hist, kernel, boundary='fill', fill_value=0)
        sky_hist_rem = convolve_fft(sky_hist_rem, kernel, boundary='fill', fill_value=0)
    
    # normalize time
    hit_per_str = 'px'
    if norm_time:
        total_time = sim._sky_pattern.scan_duration.value
        sky_hist = sky_hist/total_time
        sky_hist_rem = sky_hist_rem/total_time
        hit_per_str = 'px/s'

    fig = plt.figure('hitmap', figsize=(15, 10))

    # --- HISTOGRAMS ---

    # reference FoV
    
    # Combined Histogram
    ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=3, fig=fig)
    hist_comb = sky_hist + sky_hist_rem
    pcm = ax1.imshow(hist_comb.T, extent=[-max_pixel_deg, max_pixel_deg, -max_pixel_deg, max_pixel_deg], vmin=0, vmax=total_max, interpolation='nearest', origin='lower')
    ax1.set_aspect('equal', 'box')
    ax1.set(xlabel='x offset (deg)', ylabel='y offset (deg)')
    ax1.set_title(f'Total hits/{hit_per_str}')

    #ax1.add_patch(copy.copy(field))
    ax1.axvline(x=0, c='black')
    ax1.axhline(y=0, c='black')

    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("bottom", size="3%", pad=0.5)
    fig.colorbar(pcm, cax=cax1, orientation='horizontal')

    # Kept Histogram
    ax2 = plt.subplot2grid((4, 4), (0, 1), rowspan=3, fig=fig, sharex=ax1, sharey=ax1)
    pcm = ax2.imshow(sky_hist.T, extent=[-max_pixel_deg, max_pixel_deg, -max_pixel_deg, max_pixel_deg], vmin=0, vmax=kept_max, interpolation='nearest', origin='lower')
    ax2.set_aspect('equal', 'box')
    ax2.set(xlabel='x offset (deg)', ylabel='y offset (deg)')
    ax2.set_title(f'Kept hits/{hit_per_str}')

    #ax2.add_patch(copy.copy(field))
    ax2.axvline(x=0, c='black')
    ax2.axhline(y=0, c='black')

    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("bottom", size="3%", pad=0.5)
    fig.colorbar(pcm, cax=cax2, orientation='horizontal')

    # Removed Histogram
    ax3 = plt.subplot2grid((4, 4), (0, 2), rowspan=3, fig=fig, sharex=ax1, sharey=ax1)
    pcm = ax3.imshow(sky_hist_rem.T, extent=[-max_pixel_deg, max_pixel_deg, -max_pixel_deg, max_pixel_deg], vmin=0, vmax=rem_max, interpolation='nearest', origin='lower')
    ax3.set_aspect('equal', 'box')
    ax3.set(xlabel='x offset (deg)', ylabel='y offset (deg)')
    ax3.set_title(f'Removed hits/{hit_per_str}')

    #ax3.add_patch(copy.copy(field))
    ax3.axvline(x=0, c='black')
    ax3.axhline(y=0, c='black')

    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("bottom", size="3%", pad=0.5)
    fig.colorbar(pcm, cax=cax3, orientation='horizontal')

    # --- DETECTOR ELEMENTS AND PATH ----

    x_mod = sim._module['x']*pixel_size
    y_mod = sim._module['y']*pixel_size
    rot0 = sim._telescope_pattern.rot_angle[0].to(u.rad).value

    ax4 = plt.subplot2grid((4, 4), (0, 3), rowspan=3, sharex=ax1, sharey=ax1, fig=fig)
    ax4.scatter(sim._sky_pattern.x_coord.value, sim._sky_pattern.y_coord.value, color='red', s=0.01, alpha=0.1)
    ax4.scatter(x_mod*cos(rot0) - y_mod*sin(rot0), x_mod*sin(rot0) + y_mod*cos(rot0), s=0.05)

    ax4.set_aspect('equal', 'box')
    ax4.set(xlabel='x offset (deg)', ylabel='y offset (deg)')
    ax4.set_title('Initial Pixel Positions')
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes("bottom", size="3%", pad=0.5)
    cax4.axis('off')

    # --- BIN PLOTS ---

    x_edges = np.linspace(-max_pixel, max_pixel, 2*max_pixel, endpoint=False)*pixel_size
    y_edges = np.linspace(-max_pixel, max_pixel, 2*max_pixel, endpoint=False)*pixel_size

    # bin line plot (#1)
    ax5 = plt.subplot2grid((4, 4), (3, 0), colspan=2, fig=fig)
    bin_index = max_pixel
    bin_edge = x_edges[bin_index]
    y_values = sky_hist[bin_index]
    y_values_rem = sky_hist_rem[bin_index]

    ax5.plot(y_edges, y_values, label='Kept hits', drawstyle='steps')
    ax5.plot(y_edges, y_values_rem, label='Removed hits', drawstyle='steps')
    ax5.plot(y_edges, y_values + y_values_rem, label='Total hits', drawstyle='steps', color='black')

    #if self.scan.scan_type == 'pong':
    #    ax5.axvline(x=-width/2, c='r')
    #    ax5.axvline(x=width/2, c='r') 

    ax5.set(ylabel=f'Hits/{hit_per_str}', xlabel='y offset (deg)', ylim=(0, total_max))
    ax5.set_title(f'Hit count in x={round(bin_edge, 3)}', fontsize=12)
    ax5.legend(loc='upper right')

    # bin line plot (#2)
    ax6 = plt.subplot2grid((4, 4), (3, 2), colspan=2, fig=fig, sharex=ax5, sharey=ax5)
    bin_index = round(max_pixel/2)
    bin_edge = x_edges[bin_index]
    y_values = sky_hist[bin_index]
    y_values_rem = sky_hist_rem[bin_index]

    ax6.plot(y_edges, y_values, label='Kept hits', drawstyle='steps')
    ax6.plot(y_edges, y_values_rem, label='Removed hits', drawstyle='steps')
    ax6.plot(y_edges, y_values + y_values_rem, label='Total hits', drawstyle='steps', color='black')

    ax6.set(ylabel=f'Hits/{hit_per_str}', xlabel='y offset (deg)', ylim=(0, total_max))
    ax6.set_title(f'Hit count in x={round(bin_edge, 3)}) bin', fontsize=12)
    ax6.legend(loc='upper right')

    fig.tight_layout()

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()

def pxan_det(sim, norm_pxan=False, norm_time=False, path=None, show_plot=True):
    """
    Parameters
    ---------------------------------------------
    sim : Simulation
        A simulation object. 
    norm_pxan : bool, default False
        Average the number of hits by the number of pixels during pixel analysis. 
    norm_time : bool, default False
        True for hits/px/sec. False for hits/px.
    path : str or None, default None
        If not None, saves the image to the file path. 
    show_plot : bool, default True
        Whether to display the resulting figure.
    """

    assert(sim.pxan)

    det_hist = sim.det_hist
    det_hist_rem = sim.det_hist_rem
    max_hits = math.ceil(max(det_hist + det_hist_rem))

    # normalize time
    hit_per_str = 'px'
    if norm_time:
        total_time = sim._sky_pattern.scan_duration.value
        det_hist = det_hist/total_time
        det_hist_rem = det_hist_rem/total_time
        hit_per_str = 'px/s'

    #normaize pixel analysis
    if norm_pxan:
        det_hist = det_hist/sim.num_pxan
        det_hist_rem = det_hist_rem/sim.num_pxan

    fig_det = plt.figure(1, figsize=(15, 10))

    # --- DETECTOR HITMAP ---

    cm = plt.cm.get_cmap('viridis')

    x_mod = sim._module['x']*sim.pixel_size
    y_mod = sim._module['y']*sim.pixel_size

    # plot detector elem (total)
    det1 = plt.subplot2grid((2, 2), (0, 0), fig=fig_det)
    sc = det1.scatter(x_mod, y_mod, c=det_hist + det_hist_rem, cmap=cm, vmin=0, vmax=max_hits, s=15)
    fig_det.colorbar(sc, ax=det1, orientation='horizontal')
    det1.set_aspect('equal', 'box')
    det1.set(xlabel='x offset (deg)', ylabel='y offset (deg)')
    det1.set_title(f'Total hits/{hit_per_str}')

    # plot detector elem (kept)
    det2 = plt.subplot2grid((2, 2), (0, 1), fig=fig_det, sharex=det1, sharey=det1)
    sc = det2.scatter(x_mod, y_mod, c=det_hist, cmap=cm, vmin=0, vmax=max_hits, s=15)
    fig_det.colorbar(sc, ax=det2, orientation='horizontal')
    det2.set_aspect('equal', 'box')
    det2.set(xlabel='x offset (deg)', ylabel='y offset (deg)')
    det2.set_title('Kept hits per pixel')

    """# scale bar
    if self.scan.scan_type == 'pong':
        spacing = round(self.scan.spacing.to(u.deg).value, 3)
        scalebar = AnchoredSizeBar(det1.transData, spacing, label=f'{spacing} deg spacing', loc=1, pad=0.5, borderpad=0.5, sep=5)
        det1.add_artist(scalebar)
        scalebar = AnchoredSizeBar(det1.transData, spacing, label=f'{spacing} deg spacing', loc=1, pad=0.5, borderpad=0.5, sep=5)
        det2.add_artist(scalebar)"""

    fig_det.tight_layout()
    
    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()


def pxan_time(sim, norm_pxan=False, norm_time=False, path=None, show_plot=True):
    assert(sim.pxan)

    time_hist = sim.time_hist
    time_hist_rem = sim.time_hist_rem

    # normalize time
    hit_per_str = 'px'
    total_time = math.ceil(sim._sky_pattern.scan_duration.value)
    if norm_time:
        time_hist = time_hist/total_time
        time_hist_rem = time_hist_rem/total_time
        hit_per_str = 'px/s'

    #normaize pixel analysis
    if norm_pxan:
        time_hist = time_hist/sim.num_pxan
        time_hist_rem = time_hist_rem/sim.num_pxan

    fig_time = plt.figure(1, figsize=(15, 10))
    times = sim._sky_pattern.time_offset.value[sim.validity_mask]
    times_rem = sim._sky_pattern.time_offset.value[~sim.validity_mask]

    # --- 1 SEC HISTOGRAMS ---

    bins_time = range(0, total_time+1, 1)

    # total
    time1_tot = plt.subplot2grid((2, 2), (0, 0), fig=fig_time)
    time1_tot.hist(np.append(times, times_rem), bins=bins_time, weights=np.append(time_hist, time_hist_rem))
    time1_tot.set(xlabel='Time Offset (s)', ylabel='# of hits', title='Total Hits')

    # kept
    time1_kept = plt.subplot2grid((2, 2), (0, 1), sharex=time1_tot, sharey=time1_tot, fig=fig_time)
    time1_kept.hist(times, bins=bins_time, weights=time_hist)
    time1_kept.set(xlabel='Time Offset (s)', ylabel='# of hits', title='Kept Hits')

    # --- 10 SEC HISTOGRAMS ---

    bins_time = range(0, total_time+10, 10)

    # total
    time2_tot = plt.subplot2grid((2, 2), (1, 0), fig=fig_time)
    time2_tot.hist(np.append(times, times_rem), bins=bins_time, weights=np.append(time_hist, time_hist_rem))
    time2_tot.set(xlabel='Time Offset (s)', ylabel='# of hits', title='Total Hits')

    # kept
    time2_kept = plt.subplot2grid((2, 2), (1, 1), sharex=time2_tot, sharey=time2_tot, fig=fig_time)
    time2_kept.hist(times, bins=bins_time, weights=time_hist)
    time2_kept.set(xlabel='Time Offset (s)', ylabel='# of hits', title='Kept Hits')

    fig_time.tight_layout()
    
    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()


# --------------------------

def plot_focal_plane(self):
    pass

def visibility(dec, max_airmass=2, min_elevation=30, max_elevation=75, location=FYST_LOC):

    fig = plt.figure('rename', figsize=(8, 10))

    ax_elev = plt.subplot2grid((3, 1), (0, 0))
    ax_airmass = plt.subplot2grid((3, 1), (1, 0), sharex=ax_elev)
    ax_rot_rate = plt.subplot2grid((3, 1), (2, 0), sharex=ax_elev)

    ax_list = (ax_elev, ax_airmass, ax_rot_rate)

    # get elevation

    hour_angle = np.linspace(-12, 12, 100, endpoint=False)
    hour_angle_rad = (hour_angle*u.hourangle).to(u.rad).value
    dec_rad = u.Quantity(dec, u.deg).to(u.rad).value
    lat_rad = location.lat.rad

    alt_rad = np.arcsin( np.sin(dec_rad)*sin(lat_rad) + np.cos(dec_rad)*cos(lat_rad)*np.cos(hour_angle_rad) )
    alt = np.degrees(alt_rad)

    # filter out points
    min_elevation = u.Quantity(min_elevation, u.deg).value
    airmass = 1/np.cos(pi - alt_rad)

    mask = (airmass < max_airmass) & (alt > min_elevation)

    alt_rad = alt_rad[mask]
    hour_angle_rad = hour_angle_rad[mask]
    alt = alt[mask]
    hour_angle = hour_angle[mask]
    airmass = airmass[mask]

    # filter out max_elevation
    max_elevation = u.Quantity(max_elevation, u.deg).value
    mask_max_el = alt < max_elevation

    # elevation plot
    ax_elev.plot(hour_angle[mask_max_el], alt[mask_max_el], color='blue')
    ax_elev.plot(hour_angle[~mask_max_el], alt[~mask_max_el], color='blue', ls='dashed')

    # airmass plot
    ax_airmass.plot(hour_angle[mask_max_el], airmass[mask_max_el], color='blue')
    ax_airmass.plot(hour_angle[~mask_max_el], airmass[~mask_max_el], color='blue', ls='dashed')

    # parallactic angle and rotation angle
    para_angle_rad = np.arctan2(
        np.sin(hour_angle_rad),
        (cos(dec_rad)*tan(lat_rad) - sin(dec_rad)*np.cos(hour_angle_rad))
    )
    rot_angle_rad = para_angle_rad + alt_rad
    rot_angle = np.degrees(rot_angle_rad)
    
    # rotation angle rate
    mask_hrang = hour_angle < 0

    center_rot_i = list(hour_angle).index( max(hour_angle[mask_hrang]) )
    low_rot = rot_angle[center_rot_i] - 180
    rot_norm = (rot_angle - low_rot)%360 + low_rot # ((value - low) % diff) + low 

    SIDEREAL_TO_UT1 = 1.002737909350795
    freq_hr = (hour_angle[1]-hour_angle[0])/SIDEREAL_TO_UT1

    rot_rate = np.diff(rot_norm, append=math.nan)/freq_hr
    ax_rot_rate.plot(hour_angle[mask_max_el & mask_hrang], rot_rate[mask_max_el & mask_hrang], color='blue')
    ax_rot_rate.plot(hour_angle[mask_max_el & ~mask_hrang], rot_rate[mask_max_el & ~mask_hrang], color='blue')
    ax_rot_rate.plot(hour_angle[~mask_max_el], rot_rate[~mask_max_el], color='blue', ls='dashed')

    max_abs_rot_rate = max(60, np.max(abs(rot_rate[mask_max_el])) )

    # handle xticks and other settings of ax

    xlabel = 'Hourangle [hours]'
    xticks = [i for i in range(-12, 13, 3)]
    xtick_labels = [f'{t + 24}h' if t < 0 else f'{t}h' for t in xticks]

    for ax in ax_list:
        ax.set(xticks=xticks, xticklabels=xtick_labels)

    ax_airmass.set_yscale('function', functions=(lambda x: np.log10(x), lambda x: 10**x))
    


    ax_airmass.set(
        title='Airmass for Visible Objects (airmass < 2) at FYST', xlabel=xlabel, 
        ylabel='Airmass', ylim=( max(1.001, 1/cos(pi/2 - radians(max_elevation)) ) , max_airmass)
    )
    ax_airmass.invert_yaxis()
    ax_airmass.axhline(1/cos(pi/2 - math.radians(75)), ls='dashed', color='black')

    ax_elev.set(
        title='Elevation for Visible Objects (airmass < 2) at FYST', xlabel=xlabel,
        ylabel='Elevation [deg]', ylim=(min_elevation, min(max_elevation, 87))
    )
    ax_elev.set_yticks(np.append(ax_elev.get_yticks(), 75))
    ax_elev.axhline(75, ls='dashed', color='black')

    ax_rot_rate.set(
        title='Field Rotation Angle Rate', xlabel=xlabel, 
        ylabel='Field Rotation Angle Rate [deg/hr]', ylim=(-max_abs_rot_rate, max_abs_rot_rate)
    )
    ax_rot_rate.axhline(15, ls=':', color='black')
    ax_rot_rate.axhline(-15, ls=':', color='black')
    if max_abs_rot_rate == 60:
        ax_rot_rate.set_yticks(range(-60, 61, 15))    
    else:
        ax_rot_rate.set_yticks(np.append(ax_rot_rate.get_yticks(), [-15, 15]))

    # secondary axis for azimuthal scale factor

    def transform(x):
        np.seterr(invalid='ignore', divide='ignore')
        new_x = 1/np.sin(np.arccos(1/x))
        mask = np.isinf(new_x)
        new_x[mask] = math.nan
        np.seterr(invalid='warn', divide='warn')
        return new_x

    def inverse(x):
        np.seterr(invalid='ignore', divide='ignore')
        new_x = 1/np.cos(np.arcsin(1/x)) 
        mask = np.isinf(new_x)
        new_x[mask] = math.nan
        np.seterr(invalid='warn', divide='warn')
        return new_x

    ax_airmass_right = ax_airmass.secondary_yaxis('right', functions=(transform, inverse))
    ax_airmass_right.set(ylabel='Azimuthal Scale Factor')
    ax_airmass_right.set_yticks([1.2, 1.5, 2, 2.5, 3, 4])

    ax_elev_right = ax_elev.secondary_yaxis('right', functions=( lambda x: 1/np.cos(np.radians(x)), lambda x: np.degrees(np.arccos(1/x)) ))
    ax_elev_right.set(ylabel='Azimuthal Scale Factor')
    ax_elev_right.set_yticks([1.2, 2, 3, 4, 5, 6, 15])

    # final touchups to axis

    ax_airmass.legend(loc='lower right')
    ax_elev.legend(loc='lower right')

    ax_rot_rate.legend(loc='lower right')

    for ax in ax_list:
        ax.grid()

    fig.tight_layout()

    plt.show()

