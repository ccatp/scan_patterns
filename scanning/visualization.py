from math import pi, sin, cos, tan, radians
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.utils import isiterable
import astropy.units as u

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

