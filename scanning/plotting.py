import matplotlib.pyplot as plt
from math import pi, sin, cos, radians, degrees
import numpy as np

def instrument_config(instrument):

    fig = plt.figure()
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
    ax.set(xlabel='[deg]', ylabel='[deg]')
    plt.show()

def telescope_kinematics(telescope_pattern, module=None):

    # COORDINATES

    fig_coord = plt.figure('coordinates')

    # azmiuth vs elevation
    ax_coord = plt.subplot2grid((2, 1), (0, 0))
    ax_coord.plot(telescope_pattern.az_coord.value, telescope_pattern.alt_coord.value)
    ax_coord.set_aspect('equal', 'box')
    ax_coord.set(xlabel='Azimuth [deg]', ylabel='Elevation [deg]', title=f'Path')
    ax_coord.grid()

    # time vs. azmiuth/elevation
    ax_coord_time = plt.subplot2grid((2, 1), (1, 0))
    start_az = telescope_pattern.az_coord.value[0]
    start_alt = telescope_pattern.alt_coord.value[0]

    ax_coord_time.plot(telescope_pattern.time_offset.value, (telescope_pattern.az_coord.value - start_az)*cos(radians(start_alt)), label=f'AZ from {round(start_az, 2)} deg')
    ax_coord_time.plot(telescope_pattern.time_offset.value, telescope_pattern.alt_coord.value - start_alt, label=f'EL from {round(start_alt, 2)} deg')
    ax_coord_time.legend(loc='upper right')
    ax_coord_time.set(xlabel='Time Offset [s]', ylabel='Position [deg]', title=f'Time vs. Position')
    ax_coord_time.grid()

    fig_coord.tight_layout()

    # MOTION

    fig_motion = plt.figure('motion')

    # velocity
    ax_vel = plt.subplot2grid((3, 1), (0, 0))
    ax_vel.plot(telescope_pattern.time_offset.value, telescope_pattern.vel.value, label='Total', c='black', ls='dashed', alpha=0.25)
    ax_vel.plot(telescope_pattern.time_offset.value, telescope_pattern.az_vel.value, label='AZ')
    ax_vel.plot(telescope_pattern.time_offset.value, telescope_pattern.alt_vel.value, label='EL')
    ax_vel.legend(loc='upper right')
    ax_vel.set(xlabel='Time offset [s]', ylabel='Velocity [deg/s]', title=f'Time vs. Velocity')
    ax_vel.grid()

    # acceleration
    ax_acc = plt.subplot2grid((3, 1), (1, 0), sharex=ax_vel)
    ax_acc.plot(telescope_pattern.time_offset.value, telescope_pattern.acc.value, label='Total', c='black', ls='dashed', alpha=0.25)
    ax_acc.plot(telescope_pattern.time_offset.value, telescope_pattern.az_acc.value, label='AZ')
    ax_acc.plot(telescope_pattern.time_offset.value, telescope_pattern.alt_acc.value, label='EL')
    ax_acc.legend(loc='upper right')
    ax_acc.set(xlabel='Time offset [s]', ylabel='Acceleration [deg/s^2]', title=f'Time vs. Acceleration')
    ax_acc.grid()

    # jerk
    ax_jerk = plt.subplot2grid((3, 1), (2, 0), sharex=ax_vel)
    ax_jerk.plot(telescope_pattern.time_offset.value, telescope_pattern.jerk.value, label='Total', c='black', ls='dashed', alpha=0.25)
    ax_jerk.plot(telescope_pattern.time_offset.value, telescope_pattern.az_jerk.value, label='AZ')
    ax_jerk.plot(telescope_pattern.time_offset.value, telescope_pattern.alt_jerk.value, label='EL')
    ax_jerk.legend(loc='upper right')
    ax_jerk.set(xlabel='Time Offset (s)', ylabel='Jerk [deg/s^2]', title=f'Time vs. Jerk')
    ax_jerk.grid()

    fig_motion.tight_layout()

    plt.show()

def sky_kinematics(module=None):
    pass

def plot_focal_plane(self):
    pass