import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import scipy.ndimage as ndimage
import pandas as pd
import re
import numpy as np
import time
import sys

np.set_printoptions(threshold=np.inf)
# If file does not work, check that read algorithm works first!
# Simply change read algorithm if more PRI.STA options provided

# WORKS FOR:
# read_file = "cycle01_KINF.out"
# read_file = "cycle01_VOI.out"
read_file = "cycle01_RPD.out"


seeking_strings = ["ROW  1/01", "ROW  2/02", "ROW  3/03", "ROW  4/04", "ROW  5/05", "ROW  6/06", "ROW  7/07",
                   "ROW  8/08", "ROW  9/09", "ROW 10/10", "ROW 11/11", "ROW 12/12", "ROW 13/13", "ROW 14/14",
                   "ROW 15/15"]

# dec_behavior = "\d+\."
dec_behavior = "\d+\.\d+"


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

burn_up = burnup_read(read_file)


def boundary_read(file):
    string = "Axial Nodal Boundaries (cm)"

    with open(file, 'r+') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if string in lines[i]:
                num_lines = []
                for k in range(27):

                    nex = lines[i + k + 2]
                    num_line = (re.findall("\d+\.\d+", nex))
                    num_line = ([float(x) for x in num_line])

                    num_lines.append(num_line)

                arr = np.array(num_lines)

                return arr

array_1 = boundary_read(file=read_file)
array_1[-1:] = -1*array_1[-1]
array_1 = array_1 + 15.24
array_1 = array_1[-24:]
# print(array_1)
# Make last entry negative, Normalize, Reverse, Take only first 24


def data_read(strings, file):
    read_matrix = []

    with open(file, 'r+') as f:
        lines = f.readlines()

        for i in range(len(lines)):
            line = lines[i]

            for s in range(len(strings)):

                if strings[s] in lines[i]:

                    # print(i, lines[i])

                    # even row
                    if (s + 1) % 2 == 0 and s != 1 or s == 0:

                        for k in range(len(array_1)):

                            nex = lines[i + k + 3]

                            # Get float numbers later: "\d+\.\d+"
                            num_line = (re.findall(dec_behavior, nex))
                            num_line = ([float(x) for x in num_line])

                            if len(num_line) > 2:
                                # print(num_line)

                                # Dimension of Core (must be odd)
                                length = 17
                                edge_len = (length - len(num_line)) / 2
                                temp = np.zeros(length)
                                temp[int(edge_len):-int(edge_len)] = num_line
                                num_line = temp

                                read_matrix.append(num_line)

                    else:

                        for k in range(len(array_1)):

                            nex = lines[i + k + 1]

                            # Get float numbers later: "\d+\.\d+"
                            num_line = (re.findall(dec_behavior, nex))
                            num_line = ([float(x) for x in num_line])

                            if len(num_line) > 2:
                                # print(num_line)

                                # Dimension of Core (must be odd)
                                length = 17
                                edge_len = (length - len(num_line)) / 2
                                temp = np.zeros(length)
                                temp[int(edge_len):-int(edge_len)] = num_line
                                num_line = temp

                                read_matrix.append(num_line)

    ht_24 = []
    ht_23 = []
    ht_22 = []
    ht_21 = []
    ht_20 = []
    ht_19 = []
    ht_18 = []
    ht_17 = []
    ht_16 = []
    ht_15 = []
    ht_14 = []
    ht_13 = []
    ht_12 = []
    ht_11 = []
    ht_10 = []
    ht_09 = []
    ht_08 = []
    ht_07 = []
    ht_06 = []
    ht_05 = []
    ht_04 = []
    ht_03 = []
    ht_02 = []
    ht_01 = []

    # print(np.shape(read_matrix))
    # for entry in read_matrix:
    #     print(entry)

    all_arr = [ht_24,
               ht_23,
               ht_22,
               ht_21,
               ht_20,
               ht_19,
               ht_18,
               ht_17,
               ht_16,
               ht_15,
               ht_14,
               ht_13,
               ht_12,
               ht_11,
               ht_10,
               ht_09,
               ht_08,
               ht_07,
               ht_06,
               ht_05,
               ht_04,
               ht_03,
               ht_02,
               ht_01]

    for i in range(24):
        for j in range(0, len(read_matrix), 24):
            all_arr[i].append(read_matrix[j + i])

    for l in range(len(all_arr)):
        all_arr[l] = np.array(np.vsplit(np.array(all_arr[l]), 74))

    new_array = []
    for a in range(len(all_arr)):
        for b in range(len(all_arr[a])):

            # new = np.pad(all_arr[a][b], ((1, 1), (0, 0)), mode='constant', constant_values=0)
            new = all_arr[a][b][:, 1:-1]
            # print(pd.DataFrame(new))
            new_array.append(new)

    new_array = np.array(np.vsplit(np.array(new_array), 74))
    # print(np.shape(new_array))

    return new_array


full_arr = data_read(strings=seeking_strings, file=read_file)
# Reads File for strings - Output Full Data Array ([Burnup][height][layout])

# for a in range(len(full_arr)):
#     for b in range(len(full_arr[a])):
#         print(pd.DataFrame(full_arr[a][b]))


def Modded_Full_Layout(array, resolution):
    new_array = []
    for b in range(len(array)):
        for c in range(len(array[0])):
            new = np.kron(array[b][c], np.ones((resolution, resolution), dtype=float))
            temp = np.zeros(np.array(new).shape + np.array([2]), np.array(new).dtype)
            temp[1:-1, 1:-1] = new
            new = temp

            new_array.append(new)

    new_array = np.array(np.vsplit(np.array(new_array), 74))

    return new_array


def Val_Append(arr, value, num_of_vals):
    for a in range(num_of_vals):
        arr.append(value)
    for b in range(8-num_of_vals):
        arr.append(0)
    return arr


def Fill(value):
    other = []
    new = []
    i = 1
    for x in [8, 8, 7, 7, 6, 5, 4, 2]:
        Val_Append(other, value, x)
        new = np.array(other).reshape(i, 8)
        i += 1
    return new


def Mirror(seq):
    output = list(seq[::-1])
    output.extend(seq[1:])
    return output


def Full_Layout(array, resolution):
    f_matrices = []
    for b in range(len(array)):

        matrix = Mirror([Mirror(sublist) for sublist in array[b]])
        matrix = np.kron(matrix, np.ones((resolution, resolution), dtype=float))

        temp = np.zeros(np.array(matrix).shape + np.array([2]), np.array(matrix).dtype)
        temp[1:-1, 1:-1] = matrix
        matrix = temp

        f_matrices.append(matrix)

    f_matrices = np.array(f_matrices)
    f_matrices = np.array(np.vsplit(f_matrices, len(burn_up)))

    return f_matrices

res = 5
color_matrices = Modded_Full_Layout(full_arr, resolution=res)

###############################################################

val_max = 0
for a in range(len(color_matrices)):
    for b in range(len(color_matrices[a])):
        if val_max < np.max(color_matrices[a][b]):
            val_max = np.max(color_matrices[a][b])

val_min = val_max
for a in range(len(color_matrices)):
    for b in range(len(color_matrices[a])):
        if val_min > np.min(color_matrices[a][b]):
            val_min = np.min(color_matrices[a][b])


###############################################################

filled_arr = []
for b in range(len(burn_up)):
    for a in range(len(array_1)):
        filled_arr.append(Fill(float(array_1[a])))

height_matrices = Full_Layout(array=filled_arr, resolution=res)

print(np.shape(height_matrices))
print(np.shape(color_matrices))

# for a in range(len(color_matrices)):
#     for b in range(len(color_matrices[a])):
#         print(pd.DataFrame(color_matrices[a][b]))

# for i in range(len(height_matrices)):
#     for j in range(len(height_matrices[i])):
#         print(pd.DataFrame(height_matrices[j][i]))

measure_matrix = np.array(color_matrices[0][0])
conv_arr_width = measure_matrix.shape[0]
conv_arr_length = measure_matrix.shape[0]


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

name = []
###################################################################

tls.set_credentials_file(username='sterlingbutters', api_key='2dc5zzdbva')

stream_id = tls.get_credentials_file()['stream_ids']
token = stream_id[-2]

stream_id = dict(token=token)

z_init = np.zeros(100).reshape((10, 10))
z = z_init

surface = go.Surface(z=z,
                     stream=stream_id,
                     cmin=val_min,
                     cmax=val_max,
                     colorscale='Viridis',
                     opacity=.99,
                     name="Start",
                     hoverinfo="all"
                     )

data = [surface]

layout = go.Layout(
     title='Distribution Through Height of the Core w/ Burnup',
     scene=dict(
        zaxis=dict(range=[.1, 400])),
     width=700,
     height=1000,)
     # margin=dict(r=20, l=10,
     #             b=10, t=10))

fig = go.Figure(data=data, layout=layout)
plot_url = py.iplot(fig, filename='', auto_open=True)

##################################################################

s = py.Stream(stream_id=token)
s.open()

time.sleep(5)

sleep_time = .05

i = 35
while i < len(height_matrices):

    for b in range(len(array_1) - 1):

        height_between = interp(height_matrices[i][b], height_matrices[i][b+1], numpoints=10, order=2)
        color_between = interp(color_matrices[i][b], color_matrices[i][b+1], numpoints=10, order=2)
        burnup_text = [['Burnup: {} GWd/MT'.format(burn_up[i]) for d in range(conv_arr_width)] for e in range(conv_arr_length)]

        r = 0
        for r in range(len(color_between)):

            s.write(go.Surface(z=height_between[r],
                               surfacecolor=color_between[r],
                               # text=color_between[r],
                               text=burnup_text,
                               name="",
                               hoverinfo="all"
                               ))

            time.sleep(sleep_time)

        print('Axial Step = {}/{}'.format(b, len(array_1) - 1))

    print('Burnup Step = {}/{}'.format(i, len(burn_up)))
    print()

    i += 1
    time.sleep(1)

# s.close()
