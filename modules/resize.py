import PySimpleGUI as sg
import numpy as np
import math


def open_window_resize(np_image):

    layout = [[sg.Text("height"),sg.Input('',enable_events=True, key='-HEIGHTINPUT-'),sg.Text("px")],
    [sg.Text("width"),sg.Input('',enable_events=True, key='-WIDTHINPUT-'),sg.Text("px")],
    [sg.Text("constrained"),sg.Checkbox('', key='s1')],
    [sg.Button('NN'),sg.Button('Bilinear'),sg.Button('Cancel')]
    ]

    window = sg.Window("Resize", layout, modal=True)

    while True:

        event, values = window.read()

        if event == 'Cancel' or event == sg.WIN_CLOSED:
            window.close()
            return np_image
            break

        if event == 'NN':
            if values['s1']==True:
                h , w = constrained(np_image,values['-HEIGHTINPUT-'],values['-WIDTHINPUT-'])
                np_image = resizeNN(np_image,h,w)
            else:
                h = values['-HEIGHTINPUT-']
                w = values['-WIDTHINPUT-']
                np_image = resizeNN(np_image,h,w)

            window.close()
            return np_image
            break
        if event == 'Bilinear':
            if values['s1']==True:
                h , w = constrained(np_image,values['-HEIGHTINPUT-'],values['-WIDTHINPUT-'])
                np_image = resizebil(np_image,h,w)
            else:
                h = values['-HEIGHTINPUT-']
                w = values['-WIDTHINPUT-']
                np_image = resizebil(np_image,values['-HEIGHTINPUT-'],values['-WIDTHINPUT-'])

            window.close()
            return np_image
            break
    window.close()


def constrained(image,vh,vw):
    h,w,c = image.shape

    if not vh:
        vw = int(vw)
        temp = vw/  w
        neww = vw
        newh = round(h * temp)
    else:
        vh = int(vh)
        temp = vh/  h
        newh = vh
        neww = round(w * temp)

    return newh,neww


def resizeNN(image,h,w):
    h = int(h)
    w = int(w)
    new_image = np.zeros((int(h), int(w), 3), np.uint8)

    if len(image.shape) ==2:
        oldh,oldw = image.shape
    else:
        oldh,oldw,oldc = image.shape


    frac_w = oldw / w
    frac_h = oldh / h

    curr_h = 0
    for i in range(h):
        curr_w = 0
        for x in range(w):
            new_image[i][x] = image[math.floor(curr_h)][math.floor(curr_w)]
            curr_w +=frac_w
        curr_h +=frac_h
    return new_image


def resizebil(image,h,w):
    h = int(h)
    w = int(w)

    if len(image.shape) ==2:
        oldh,oldw = image.shape
    else:
        oldh,oldw,oldc = image.shape

    new_image = np.zeros((h, w, 3), np.uint8)
    frac_w = oldw / w
    frac_h = oldh / h
    for i in range(h):
    	for x in range(w):
    		curr_x = i * frac_h
    		y = x * frac_w

    		x_floor = math.floor(curr_x)
    		x_ceil = min( oldh - 1, math.ceil(curr_x))
    		y_floor = math.floor(y)
    		y_ceil = min(oldw - 1, math.ceil(y))

    		if (x_ceil == x_floor) and (y_ceil == y_floor):
    			q = image[int(curr_x), int(y), :]
    		elif (x_ceil == x_floor):
    			q1 = image[int(curr_x), int(y_floor), :]
    			q2 = image[int(curr_x), int(y_ceil), :]
    			q = q1 * (y_ceil - y) + q2 * (y - y_floor)
    		elif (y_ceil == y_floor):
    			q1 = image[int(x_floor), int(y), :]
    			q2 = image[int(x_ceil), int(y), :]
    			q = (q1 * (x_ceil - curr_x)) + (q2	 * (curr_x - x_floor))
    		else:
    			v1 = image[x_floor, y_floor, :]
    			v2 = image[x_ceil, y_floor, :]
    			v3 = image[x_floor, y_ceil, :]
    			v4 = image[x_ceil, y_ceil, :]

    			q1 = v1 * (x_ceil - curr_x) + v2 * (curr_x - x_floor)
    			q2 = v3 * (x_ceil - curr_x) + v4 * (curr_x - x_floor)
    			q = q1 * (y_ceil - y) + q2 * (y - y_floor)

    		new_image[i][x] = q
    return new_image
