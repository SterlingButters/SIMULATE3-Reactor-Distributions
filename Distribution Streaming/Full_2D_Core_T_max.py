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
file = "cycle01.out"

# string = "PRI.STA 2RPF  - Assembly 2D Ave RPF - Relative Power Fraction"
# string = "PRI.STA 2EXP  - Assembly 2D Ave EXPOSURE  - GWD/T"
# string = "PIN.EDT 2PIN  - Peak Pin Power:              Assembly 2D"     # Check
# string = "PRI.STA 2KIN  - Assembly 2D Ave KINF - K-infinity"
# string = "PRI.STA 2DEN  - Assembly 2D Ave DEN - Moderator Density g/cc"
# string = "PRI.STA 2VOI  - Assembly 2D Ave VOI - Inchannel Void Fraction"
string_RPD = " PRI.STA 2RPD  - Assembly 2D Ave RPD - Relative Power Density kW/ft"
string_T = " PRI.STA 2TFU  - Assembly 2D Ave TFU - Fuel Temperature K"
# string = " PRI.STA 2FLX  - Assembly 2D Ave FLX - Group 1 Flux                      REAL VALUE * 1.0E-14 = EDIT VALUE"
# string = " PRI.STA 2FLX  - Assembly 2D Ave FLX - Group 2 Flux                      REAL VALUE * 1.0E-14 = EDIT VALUE"
# string = " PRI.STA 2FLO  - Assembly 2D ACTIVE FLOW, LB/HR                      REAL VALUE * 1.0E-05 = EDIT VALUE"


def read_data_nodec(string):
    matrices = []

    with open(file, 'r+') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if string in lines[i]:

                num_lines = []

                for k in range(15):

                    nex = lines[i + k + 2]
                    num_line = (re.findall("\d+\.", nex))
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

    return matrices


def read_data(string):
    matrices = []

    with open(file, 'r+') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if string in lines[i]:

                num_lines = []

                for k in range(15):

                    nex = lines[i + k + 2]
                    num_line = (re.findall("\d+\.\d+", nex))
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

    return matrices


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

T_avg = read_data_nodec(string_T)
RPD = read_data(string_RPD)

k_f = 5

T_max = []
T_diff = []
for b in range(len(T_avg)):
    # From Todreas and Kazimi
    T_max.append(T_avg[b] - RPD[b]/(8 * np.pi * k_f))
    T_diff.append(RPD[b]/(8 * np.pi * k_f))

# for b in range(len(T_max)):
#     print(pd.DataFrame(T_max[b]))

measure_matrix = np.array(T_max[0])
conv_arr_width = measure_matrix.shape[0]
conv_arr_length = measure_matrix.shape[1]

maxes = []
# for m in range(len(T_max)):
#     c_max = np.amax(np.array(T_max[m]))
for m in range(len(T_diff)):
    c_max = np.amax(np.array(T_diff[m]))
    maxes.append(c_max)
c_max = np.max(np.array(maxes))

print('Burnup Steps:', len(T_max))

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
token1 = stream_id[-1]
token2 = stream_id[-2]


stream_id1 = dict(token=token1)
stream_id2 = dict(token=token2)

z_init = np.zeros(100).reshape((10, 10))
z = z_init

surface1 = go.Surface(z=z,
                      stream=stream_id1,
                      zmin=-c_max,
                      zmax=c_max,
                      name="T_avg",
                      opacity=1,
                      colorscale='Viridis',
                      hoverinfo="all"
                     )

surface2 = go.Surface(z=z,
                      stream=stream_id2,
                      zmin=-c_max,
                      zmax=c_max,
                      name="T_max",
                      opacity=1,
                      colorscale='Viridis',
                      # Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis
                      hoverinfo="all"
                      )

data = [surface1, surface2]

layout = go.Layout(
     title='Reactor Distribution',
     scene=dict(
        zaxis=dict(range=[0, c_max])),
     width=700,
     margin=dict(r=20, l=10,
                 b=10, t=10))

fig = go.Figure(data=data, layout=layout)
plot_url = py.iplot(fig, filename='', auto_open=True)

##################################################################

s1 = py.Stream(stream_id=token1)
s2 = py.Stream(stream_id=token2)

s1.open()
s2.open()

sleep_time = .05

i = 0
while True:

    between_T_avg = interp(T_avg[i], T_avg[i+1], 100, 2)
    between_T_max = interp(T_max[i], T_max[i+1], 100, 2)
    between_T_diff = interp(T_diff[i], T_diff[i+1], 100, 2)
    burnup_text = [['Burnup: {} GWd/MT'.format(burn_up[i]) for k in range(conv_arr_width)] for j in range(conv_arr_length)]

    r = 0
    for r in range(len(between_T_max)):
        s1.write(go.Surface(z=between_T_avg[r],
                           text=burnup_text,
                           hoverinfo="all"
                           ))

        # s2.write(go.Surface(z=between_T_max[r],
        #                     text=burnup_text,
        #                     hoverinfo="all"
        #                     ))
        
        time.sleep(sleep_time)

    i += 1

    print('i = {}/{}'.format(i, len(T_max)))

    # if i == len(matrices)-3:
    #     i = 0
    #     b = 0

    if i == len(T_max)-2:
        s1.close()
        # s2.close()
