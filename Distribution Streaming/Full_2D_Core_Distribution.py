import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import scipy.ndimage as ndimage
from random import uniform
import pandas as pd
import re
import numpy as np
import time
import sys

# np.set_printoptions(threshold=np.inf)

resolution = 5


def mirror(seq):
    output = list(seq[::-1])
    output.extend(seq[1:])
    return output

##################################################################

# string = "PRI.STA 2RPF  - Assembly 2D Ave RPF - Relative Power Fraction"
# string = "PRI.STA 2EXP  - Assembly 2D Ave EXPOSURE  - GWD/T"
# string = "PIN.EDT 2PIN  - Peak Pin Power:              Assembly 2D"     # Check
# string = "PRI.STA 2KIN  - Assembly 2D Ave KINF - K-infinity"
# string = "PRI.STA 2DEN  - Assembly 2D Ave DEN - Moderator Density g/cc"
string = "PRI.STA 2VOI  - Assembly 2D Ave VOI - Inchannel Void Fraction"
# string = "PRI.STA 2RPD  - Assembly 2D Ave RPD - Relative Power Density kW/ft"
# string = "PRI.STA 2TFU  - Assembly 2D Ave TFU - Fuel Temperature K"
# string = "PRI.STA 2FLX  - Assembly 2D Ave FLX - Group 1 Flux                      REAL VALUE * 1.0E-14 = EDIT VALUE"
# string = "PRI.STA 2FLX  - Assembly 2D Ave FLX - Group 2 Flux                      REAL VALUE * 1.0E-14 = EDIT VALUE"
# string = "PRI.STA 2FLO  - Assembly 2D ACTIVE FLOW, LB/HR                      REAL VALUE * 1.0E-05 = EDIT VALUE"

# title = 'Assembly 2D Ave Relative Power Density kW/ft'
# title = 'Assembly 2D Ave Moderator Density g/cc'
title = 'Assembly 2D Ave Inchannel Void Fraction'
# title = 'Assembly 2D Ave Fuel Temperature K'
# title = 'Assembly 2D Active Flow, lb/hr'


file = "cycle01.out"


# dec_behavior = "\d+\.\d+"
dec_behavior = "\d+\."

matrices = []

with open(file, 'r+') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        if string in lines[i]:

            num_lines = []

            for k in range(15):

                nex = lines[i + k + 2]
                num_line = (re.findall(dec_behavior, nex))
                num_line = ([float(x) for x in num_line])

                if len(num_line) > 0:
                    while len(num_line) < 15:
                        num_line.insert(0, 0.0)
                        num_line.append(0.0)

                    num_lines.append(num_line)

            num_lines = np.array(num_lines)
            matrix = np.kron(num_lines, np.ones((resolution, resolution), dtype=float))

            # Surround zeros here
            temp = np.zeros(np.array(matrix).shape + np.array([2]), np.array(matrix).dtype)
            temp[1:-1, 1:-1] = matrix
            matrix = temp
            matrices.append(matrix)


def burnup_read(file):
    string = "  Case     Step"
    string_2 = "1S I M U L A T E - 3"
    num_lines = []
    with open(file, 'r+') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if string in lines[i]:

                a = 0
                b = 0
                while b == 0:
                    nex = lines[i + a + 3]

                    if string_2 not in nex:
                        a += 1
                    else:
                        b = 1
                    num_lines.append(nex)

    burn_up = []
    for a in range(len(num_lines)):
        num_line = (re.findall("\d+\.\d+", num_lines[a]))
        num_line = ([float(x) for x in num_line])
        if len(num_line) > 3:
            burn_up.append(num_line[0])

    burn_up = np.array(burn_up)

    return burn_up
burn_up = burnup_read(file)
# Fix "outer scope warnings"

measure_matrix = np.array(matrices[0])
conv_arr_width = measure_matrix.shape[0]
conv_arr_length = measure_matrix.shape[1]

maxes = []
for m in range(len(matrices)):
    c_max = np.amax(np.array(matrices[m]))
    maxes.append(c_max)
c_max = np.max(np.array(maxes))

print('Burnup Steps:', len(matrices))

##################################################################
#
# matrices = []
# for p in range(100):
#     matrix = []
#     for x in range(arr_width):
#         row = []
#         for y in range(arr_length):
#             row.append(uniform(-1, 1))
#         matrix.append(row)
#
##################################################################

# given 2 arrays arr1, arr2, number of steps between arrays, and order of interpolation
# numpoints = 10
# order = 2
# arr1 = matrices[1]
# arr2 = matrices[2]


def interp(arr1, arr2, numpoints, order):

    # rejoin arr1, arr2 into a single array of shape (2, 10, 10)
    arr = np.r_['0, 3', arr1, arr2]

    # define the grid coordinates where you want to interpolate
    X, Y = np.meshgrid(np.arange(conv_arr_width), np.arange(conv_arr_length))

    k = 0
    interp_arr_vec = []
    while k <= 1:
        coordinates = np.ones((conv_arr_width, conv_arr_length))*k, X, Y
        # given arr interpolate at coordinates
        interp_arr = ndimage.map_coordinates(arr, coordinates, order=order).T
        interp_arr_vec.append(interp_arr)

        step = 1 / numpoints
        k += step

    return interp_arr_vec

##################################################################

tls.set_credentials_file(username='sterlingbutters', api_key='2dc5zzdbva')

stream_id = tls.get_credentials_file()['stream_ids']
token = stream_id[-1]

stream_id = dict(token=token)

z_init = np.zeros(100).reshape((10, 10))
z = z_init

surface = go.Surface(z=z,
                     stream=stream_id,
                     zmin=0,
                     zmax=c_max,
                     colorscale='Viridis',
                     hoverinfo="all"
                     # lighting=dict(fresnel=.2,
                     #               roughness=.5,
                     #               specular=.05,
                     #               ambient=.8,
                     #               diffuse=.8),
                     # opacity=.99,
                     )

data = [surface]

# camera = dict(
#     up=dict(x=0, y=0, z=1),
#     center=dict(x=0, y=0, z=0),
#     eye=dict(x=0.1, y=0.1, z=1)
# )

layout = go.Layout(
     title=title,
     scene=dict(
        zaxis=dict(range=[.01, c_max])),
     width=700,
     height=1000,
     margin=dict(r=100, l=100,
                 b=100, t=100))

fig = go.Figure(data=data, layout=layout)
plot_url = py.iplot(fig, filename='', auto_open=True)

##################################################################

s = py.Stream(stream_id=token)
s.open()

sleep_time = .05

i = 0
while True:

    between = interp(matrices[i], matrices[i+1], 100, 2)
    burnup_text = [['Burnup: {} GWd/MT'.format(burn_up[i]) for k in range(conv_arr_width)] for j in range(conv_arr_length)]

    r = 0
    for r in range(len(between)):
        s.write(go.Surface(z=between[r],
                           text=burnup_text,
                           hoverinfo="all"
                           ))
        time.sleep(sleep_time)

    i += 1

    print('i = {}/{}'.format(i, len(matrices)))

    # if i == len(matrices)-3:
    #     i = 0
    #     b = 0

    if i == len(matrices)-2:
        s.close()
