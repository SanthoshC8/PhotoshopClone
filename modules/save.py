import PySimpleGUI as sg
import numpy as np
from PIL import Image

def open_window_save(filename,np_image):

    layout = [[sg.Text("filename"),sg.Input(filename,enable_events=True, key='-INPUT-')],
    [sg.Button('Save'),sg.Button('Cancel')]
    ]

    window = sg.Window("Save", layout, modal=True)

    while True:
        event, values = window.read()
        if event == 'Cancel' or event == sg.WIN_CLOSED:
            window.close()
            return filename
            break
        if event == 'Save':
            temp = values['-INPUT-']
            try:
                np_image = Image.fromarray(np_image)
            except:
                np_image = Image.fromarray((np_image).astype(np.uint8))

            np_image = np_image.save(temp)
            print('saved file')
            window.close()
            return temp
            break

    window.close()
