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
