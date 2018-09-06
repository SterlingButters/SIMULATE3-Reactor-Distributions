import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
import scipy.ndimage as ndimage
import re
import numpy as np
import plotly
import pandas as pd
import sys

# np.set_printoptions(threshold=np.inf)

init_notebook_mode(connected=True)

resolution = 5


def mirror(seq):
    output = list(seq[::-1])
    output.extend(seq[1:])
    return output


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

matrices = []

# string = "PRI.STA 2RPF  - Assembly 2D Ave RPF - Relative Power Fraction   "
string = "PRI.STA 2EXP  - Assembly 2D Ave EXPOSURE  - GWD/T               "
file = "cycle1.out"
with open(file, 'r+') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        if string in lines[i]:
            num_lines = []
            for k in range(8):

                nex = lines[i + k + 2]
                num_line = (re.findall("\d+\.\d+", nex))
                num_line = ([float(x) for x in num_line])

                while len(num_line) < 8:
                    num_line.append(0.0)

                num_lines.append(num_line)

            num_lines = np.array(num_lines)
            matrix = np.kron(num_lines, np.ones((resolution, resolution), dtype=float))
            matrix = mirror([mirror(sublist) for sublist in matrix])

            # Surround zeros here
            temp = np.zeros(np.array(matrix).shape + np.array(2), np.array(matrix).dtype)
            temp[1:-1, 1:-1] = matrix
            matrix = temp
            matrices.append(matrix)

burnup = burnup_read(file)

measure_matrix = np.array(matrices[0])
conv_arr_width = measure_matrix.shape[0]
conv_arr_length = measure_matrix.shape[1]

maxes = []
for m in range(len(matrices)):
    c_max = np.amax(np.array(matrices[m]))
    maxes.append(c_max)
c_max = np.max(np.array(maxes))

# for x in range(len(matrices)):
#     print(pd.DataFrame(matrices[x]))

##################################################################

# Instantiate Data
data = [go.Surface(z=matrices[0],
                   zmax=c_max,
                   zmin=0,
                   colorscale='Viridis',
                   )]

##################################################################

# Instantiate Frames
frames = []
steps = []
for k in range(len(burnup)):
    frame_data = go.Surface(z=matrices[k])
    frame = dict(data=[frame_data], name=str(burnup[k]))
    frames.append(frame)

    slider_step = dict(args=[
            [str(burnup[k])],
            dict(frame=dict(duration=0, redraw=False),
                 mode='immediate',
                 transition={'duration': 0})
        ],
            label='{} GWd/MT'.format(burnup[k]),
            method='animate')
    steps.append(slider_step)

##################################################################

# Slider Control
sliders_dict = dict(active=0,                                       # Starting Position
                    yanchor='top',
                    xanchor='left',
                    currentvalue=dict(
                         font={'size': 20},
                         prefix='Burnup:',
                         visible=True,
                         xanchor='right'
                     ),
                    # Transition for slider button
                    transition=dict(duration=500,
                                    easing='cubic-in-out'),
                    pad={'b': 10, 't': 50},
                    len=.9,
                    x=0.1,
                    y=0,
                    steps=steps
                    )

##################################################################

# Layout
layout = dict(title=string,
              hovermode='closest',
              width=1500,
              height=1000,
              scene=dict(
                    zaxis=dict(range=[.01, c_max])),
              updatemenus=[dict(type='buttons',

                                buttons=[dict(args=[None,
                                                    dict(frame=dict(duration=500,
                                                                    redraw=False),
                                                         fromcurrent=True,
                                                         transition=dict(duration=100,
                                                                         easing='quadratic-in-out'))],
                                              label=u'Play',
                                              method=u'animate'
                                              ),

                                         # [] around "None" are important!
                                         dict(args=[[None], dict(frame=dict(duration=0,
                                                                            redraw=False),
                                                                 mode='immediate',
                                                                 transition=dict(duration=0))],
                                              label='Pause',
                                              method='animate'
                                              )
                                         ],

                                # Play Pause Button Location & Properties
                                direction='left',
                                pad={'r': 10, 't': 87},
                                showactive=True,
                                x=0.1,
                                xanchor='right',
                                y=0,
                                yanchor='top'
                                )],

              slider=dict(args=[
                            'slider.value', {
                                'duration': 1000,
                                'ease': 'cubic-in-out'
                            }
                        ],
                        # initialValue=burnup[0],           # ???
                        plotlycommand='animate',
                        # values=burnup,                    # ???
                        visible=True
                    ),
              sliders=[sliders_dict]
              )

##################################################################

figure = dict(data=data, layout=layout, frames=frames)
plotly.offline.plot(figure, filename='file.html')

#################################################################





