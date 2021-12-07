from math import pi, sin, cos, tan, radians
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.utils import isiterable
import astropy.units as u

from scanning.motion import SkyPattern, TelescopePattern
from scanning import FYST_LOC

# Modules and Instruments

def plot_module(mod, include_pol=True, path=None, show_plot=True):
    """
    Plot the pixel positions and default polarizations of a given module. 

    Parameters
    -----------------------------
    mod : Module
        A Module object. 
    path : str or None, default None
        If not None, saves the image to the file path. 
    show_plot : bool, default True
        Whether to display the resulting figure.
    include_pol : bool, default True
        Whether to show the Module's default polarizations. 
    """

    fig_mod = plt.figure('Module', figsize=(8, 8))
    ax_mod = plt.subplot2grid((1, 1), (0, 0))
    x_deg = mod.x.value
    y_deg = mod.y.value

    if include_pol:

        default_orientations = np.unique(mod.pol.value)

        # determine which colormap to use
        num_def = len(default_orientations)
        if num_def <= 9:
            cmap = plt.cm.get_cmap('Set1', num_def)
        else:
            cmap = plt.cm.get_cmap('hsv', num_def)

        # plot
        plot_mod = ax_mod.scatter(x_deg, y_deg, c=mod.pol, cmap=cmap, s=1) # FIXME make pixel sizes proportional

        # color bar
        cbar = plt.colorbar(plot_mod, fraction=0.046, pad=0.04)
        tick_locs = (default_orientations + 7.5)*(num_def-1)/num_def
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(default_orientations)
        cbar.ax.set_ylabel('Default Orientation [deg]')
    
    else:

        # plot
        ax_mod = plt.subplot2grid((1, 1), (0, 0))
        plot_mod = ax_mod.scatter(x_deg, y_deg, s=1)

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
        mod_rot = radians(value['mod_rot'])

        # get modules into correct position
        x_offset = value['dist']*cos(radians(value['theta']))
        y_offset = value['dist']*sin(radians(value['theta']))
        
        x1 = value['module'].x.value*cos(mod_rot) - value['module'].y.value*sin(mod_rot) + x_offset 
        y1 = value['module'].x.value*sin(mod_rot) + value['module'].y.value*cos(mod_rot) + y_offset 

        # apply instr rot/offset
        x2 = x1*cos(instr_rot) - y1*sin(instr_rot) + instr_offset[0]
        y2 = x1*sin(instr_rot) + y1*cos(instr_rot) + instr_offset[1]
        
        plt.scatter(x2, y2, s=0.5, label=identifier)
        
    ax.grid()
    ax.legend()
    ax.set_aspect('equal')
    ax.set(xlabel='[deg]', ylabel='[deg]', title='Instrument Configuration')

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

    # pattern is TelescopePattern
    elif isinstance(pattern, TelescopePattern):
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
    ax_coord.set(xlabel='RA [deg]', ylabel='DEC [deg]', title=f'Path')
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
    ax_coord.set(xlabel='Azimuth [deg]', ylabel='Elevation [deg]', title=f'Path')
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

def plot_focal_plane(self):
    pass
