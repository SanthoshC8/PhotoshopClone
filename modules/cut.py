import PySimpleGUI as sg
import numpy as np
from modules.crop import *

def open_window_cut(np_image):

    px = 0
    layout = [[sg.Text("cut"),
    sg.Input(px,enable_events=True, key='-INPUT-'),
    sg.Text("px from the"),
    sg.Combo(['top','bottom','left','right'],default_value='top' ,size=(20, 4),readonly=True, enable_events=True, key='-ORDER-'),


    ],
    [sg.Button('Cut'),sg.Button('Cancel')]
    ]

    window = sg.Window("Cut", layout, modal=True)

    height ,width ,c = np_image.shape

    while True:
        event, values = window.read()

        if event == 'Cancel' or event == sg.WIN_CLOSED:
            window.close()
            return np_image
            break
        if event == 'Cut':
            temp = int(values['-INPUT-'])

            if values['-ORDER-'] == 'top':
                np_image = tocrop(np_image,[0,0],[width,height-temp])
            elif values['-ORDER-'] == 'bottom':
                np_image = tocrop(np_image,[0,0+temp],[width,height])
            elif values['-ORDER-'] == 'left':
                np_image = tocrop(np_image,[0+temp,0],[width,height])
            elif values['-ORDER-'] == 'right':
                np_image = tocrop(np_image,[0,0],[width-temp,height])

            window.close()
            return np_image
            break

    window.close()
