import argparse
import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
import time
import scipy as sp

from modules.resize import *
from modules.blur import *
from modules.crop import *
from modules.cut import *
from modules.addwords import *
from modules.addimage import *
from modules.save import *
from modules.colour import *
from modules.gray import *
from modules.fill import *
from modules.brightness import *
from modules.saturation import *
from modules.flip import *
from modules.rotate import *
import sys

sys.setrecursionlimit(100000)
matplotlib.use('TkAgg')

# add flood fill,labs,selection rectangle reference



def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data


def display_image(np_image,filename):

    # Convert numpy array to data that sg.Graph can understand

    imagelist=[np_image]
    currentimage = 0

    brush_choice =  [None,None]

    image_data = np_im_to_data(np_image)

    if len(np_image.shape) ==2:
        h,w = np_image.shape
    else:
        h,w,c = np_image.shape

    height = h
    width = w

    image_column = [[sg.Graph(
        canvas_size=(700, 700),
        graph_bottom_left=(0, 0),
        graph_top_right=(700, 700),
        key='-IMAGE-',
        background_color='white',
        change_submits=True,
        drag_submits=True),


        ]]

    function_column = [

    [sg.Button('Undo'),
    sg.Button('Redo'),
    sg.Button('Save'),

    sg.Button('Exit')]
    ,
    [sg.HSeparator()],
    [sg.Combo(['selection','brush','fill'],default_value='selection'  ,size=(20, 4),readonly=True, enable_events=True, key='-MOUSE-'),

    ],[sg.Text("Brush Size:"),sg.Slider(range=(1,15),default_value=0,expand_x=True, enable_events=True,
    orientation='horizontal', key='-MOUSESIZE-'),
    sg.Text("Fill Tolerance :"),sg.Slider(range=(1,50),default_value=0,expand_x=True, enable_events=True,
    orientation='horizontal', key='-THRESHHOLD-')]
    ,
    [sg.HSeparator()],
    [


    sg.Button('', button_color=('black', "#FF0000"), key="255,0,0_colour", pad=(0,0)),
    sg.Button('', button_color=('black', "#FFA500"), key="255,165,0_colour", pad=(0,0)),
    sg.Button('', button_color=('black', "#FFFF00"), key="255,255,0_colour", pad=(0,0)),
    sg.Button('', button_color=('black', "#008000"), key="0,128,0_colour", pad=(0,0)),
    sg.Button('', button_color=('black', "#0000FF"), key="0,0,255_colour", pad=(0,0)),
    sg.Button('', button_color=('black', "#800080"), key="128,0,128_colour", pad=(0,0)),
    sg.Button('', button_color=('black', "#FFC0CB"), key="255,192,203_colour", pad=(0,0)),
    sg.Button('', button_color=('black', "#966919"), key="150,105,25_colour", pad=(0,0)),
    sg.Button('', button_color=('black', "#808080"), key="128,128,128_colour", pad=(0,0)),
    sg.Button('', button_color=('black', "#000000"), key="0,0,0_colour", pad=(0,0)),
    sg.Button('', button_color=('black', "#FFFFFF"), key="255,255,255_colour", pad=(0,0)),
    sg.Text(key='info', size=(20, 1)),
    sg.Button('Gray'),
    ],
    [sg.HSeparator()],
    [sg.Text('Brightness/Saturation:'),
    sg.Slider(range=(-30,30),default_value=0, enable_events=True,
    orientation='horizontal', key='-BRIGHT-'),
    sg.Button('Brightness'),
    sg.Button('Saturation'),
    ],



    [sg.HSeparator()]
    ,
    [
    sg.Text('Blur:'),
    sg.Slider(range=(1,15),default_value=0, enable_events=True,
    orientation='horizontal', key='-SL-'),
    sg.Button('Averaging'),
    sg.Button('Gaussian'),],
    [sg.HSeparator()],
    [
    sg.Text('Text:'),
    sg.Input('',enable_events=True, key='-WORD-',size=(20, 4)),
    sg.Text('Colour:'),
    sg.Combo(['black','white','grey','red','green','blue','yellow'],default_value='black'  ,size=(20, 4),readonly=True, enable_events=True, key='-WORDCOLOUR-'),

    sg.Button('Add Text',key='Alphabet'),
    ],
    [sg.HSeparator()],
    [sg.Button('Crop'),
    sg.Button('Cut'),
    sg.Button('Resize'),
    sg.Button('Flip'),
    sg.Combo(['vertically','horizontally','custom vertically','custom horizontally'],default_value='vertically'  ,size=(15, 4),readonly=True, enable_events=True, key='-FLIP-'),
    sg.Button('Rotate'),
    sg.Combo(['90 deg','180 deg','270 deg'],default_value='90 deg'  ,size=(15, 4),readonly=True, enable_events=True, key='-ROT-'),

    ],
    [sg.HSeparator()],
    [
    sg.Button('Addimage'),
    sg.Text('to the '),
    sg.Combo(['top','bottom','left','right','custom'],default_value='custom' ,size=(20, 4),readonly=True, enable_events=True, key='-RESIZEOP-'),
    ],

    ]
    # ----- Full layout -----
    layout = [
        [sg.Column(image_column),
         sg.VSeperator(),
         sg.Column(function_column,size=(500,700)),]
    ]


    # Create the window
    window = sg.Window('Photoshop', layout, finalize=True)


    graph = window["-IMAGE-"]

    graph.draw_image(data=image_data, location=(0, height))


    dragging = False
    start_point = [None,None]
    end_point = [None,None]
    prior_rect = None


    start_data = [None,None]
    end_data = [None,None]

    brush_count = 0

    # Event loop
    while True:

        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        if event == "-IMAGE-":  # if there's a "Graph" event, then it's a mouse
            #start_time = time.time()


            x, y = values["-IMAGE-"]
            if values['-MOUSE-'] == 'selection':
                if not dragging:
                    start_point = [x,y]
                    dragging = True
                else:
                    end_point = [x,y]
                if prior_rect:
                    graph.delete_figure(prior_rect)
                if None not in (start_point, end_point):
                    prior_rect = graph.draw_rectangle(start_point, end_point, line_color='blue')
            elif values['-MOUSE-'] == 'brush':
                start_data = [x,y]
                end_data = [x+int(values["-MOUSESIZE-"]),y+int(values["-MOUSESIZE-"])]
                if brush_choice[0] != "":
                    if brush_choice[0] == "_colour":
                        np_image = changecolour(np_image,brush_choice[1],start_data,end_data)

                    elif brush_choice[0] == "Gray":
                        np_image = togray(np_image,start_data,end_data)

                    elif brush_choice[0] == "Averaging":
                        np_image =apply_filter_to_image(np_image,brush_choice[1],start_data,end_data)




                    elif brush_choice[0] == "Gaussian":
                        np_image =apply_filter_to_image(np_image,brush_choice[1],start_data,end_data)


                    brush_count+=1
                    if brush_count == 50:
                        brush_count =0
                        # image_data = np_im_to_data(np_image)
                        #
                        # graph.draw_image(data=image_data, location=(0, height))
                    # end_time = time.time()

            elif values['-MOUSE-'] == 'fill':
                #start_time = time.time()
                start_data = [x,y]
                end_data = [x,y]
                np_image = apply_effects(np_image,start_data,brush_choice,int(values["-THRESHHOLD-"]))
                # if brush_choice[0] != "":
                #     if brush_choice[0] == "_colour":
                #         #np_image = changecolour(np_image,brush_choice[1],start_data,end_data)
                #
                #         np_image = apply_effects(np_image,start_data,brush_choice[1])
                #
                #     elif brush_choice[0] == "Gray":
                #         np_image = apply_effects(np_image,start_data,brush_choice[1])







        elif event.endswith('+UP'):  # The drawing has ended because mouse up
            if values['-MOUSE-'] == 'selection':
                info = window["info"]
                start_data = start_point
                end_data = end_point
                info.update(value=f"grabbed rectangle from {start_data[0],start_data[1]} to {end_data[0],end_data[1]}")
                start_point = [None,None]
                end_point = [None,None]  # enable grabbing a new rect
                dragging = False
            else:
                imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)

                image_data = np_im_to_data(np_image)

                graph.draw_image(data=image_data, location=(0, height))



        if event.endswith("_colour"):

            colour = [int(x) for x in event[:-7].split(",")]

            if values['-MOUSE-'] == 'brush':
                brush_choice = ['_colour',colour]

            elif values['-MOUSE-'] == 'selection':
                np_image = changecolour(np_image,colour,start_data,end_data)

                imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)

                image_data = np_im_to_data(np_image)
                graph.draw_image(data=image_data, location=(0, height))
                # start_point = [None,None]
                # end_point = [None,None]  # enable grabbing a new rect
                # dragging = False

            elif values['-MOUSE-'] == 'fill':
                brush_choice = ['_colour',colour]


        if event == "Gray":

            if values['-MOUSE-'] == 'brush':
                brush_choice = ['Gray',None]

            elif values['-MOUSE-'] == 'selection':
                print('hi')
                np_image = togray(np_image,start_data,end_data)

                imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)

                image_data = np_im_to_data(np_image)
                graph.draw_image(data=image_data, location=(0, height))

            elif values['-MOUSE-'] == 'fill':
                brush_choice = ['Gray',None]

        if event =="Undo":
            if currentimage>0:
                currentimage-=1
                np_image=imagelist[currentimage]
                image_data = np_im_to_data(np_image)
                graph.erase()
                height,w,c = np_image.shape
                graph.draw_image(data=image_data, location=(0, height))


        if event =="Redo":
            if len(imagelist)-1>currentimage:
                currentimage+=1
                np_image=imagelist[currentimage]
                image_data = np_im_to_data(np_image)
                graph.erase()
                height,w,c = np_image.shape
                graph.draw_image(data=image_data, location=(0, height))

        if event =="Crop":
            np_image = tocrop(np_image,start_data,end_data)

            imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)

            image_data = np_im_to_data(np_image)
            graph.erase()
            height,w,c = np_image.shape
            graph.draw_image(data=image_data, location=(0, height))


        if event =="Rotate":
            np_image = torotate(np_image,values['-ROT-'],start_data,end_data)

            imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)

            image_data = np_im_to_data(np_image)
            graph.erase()
            height,w,c = np_image.shape
            graph.draw_image(data=image_data, location=(0, height))


        if event =="Flip":
            np_image = toflip(np_image,values['-FLIP-'],start_data,end_data)

            imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)

            image_data = np_im_to_data(np_image)

            graph.draw_image(data=image_data, location=(0, height))

        if event =='Brightness':

            np_image =change_brightness(np_image,int(values['-BRIGHT-']),start_data,end_data)

            imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)

            image_data = np_im_to_data(np_image)


            graph.draw_image(data=image_data, location=(0, height))


        if event =='Saturation':

            np_image =change_saturation(np_image,int(values['-BRIGHT-']),start_data,end_data)

            imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)

            image_data = np_im_to_data(np_image)


            graph.draw_image(data=image_data, location=(0, height))

        if event == 'Averaging':
            if values['-MOUSE-'] == 'brush':
                brush_choice = ['Averaging',[int(values['-SL-']),'a']]

            elif values['-MOUSE-'] == 'selection':

                np_image =apply_filter_to_image(np_image,[int(values['-SL-']),'a'],start_data,end_data)

                imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)

                image_data = np_im_to_data(np_image)


                graph.draw_image(data=image_data, location=(0, height))


        if event == 'Gaussian':
            tt = gaussian2d(math.sqrt(int(values['-SL-'])*(1/3)), 2*int(values['-SL-'])+1)

            if values['-MOUSE-'] == 'brush':
                brush_choice = ['Gaussian',[int(values['-SL-']),tt]]

            elif values['-MOUSE-'] == 'selection':

                np_image =apply_filter_to_image(np_image,[int(values['-SL-']),tt],start_data,end_data)

                imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)

                image_data = np_im_to_data(np_image)


                graph.draw_image(data=image_data, location=(0, height))


        if event =="Addimage":
            np_image = addimage(np_image,values['-RESIZEOP-'],start_data,end_data)

            imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)

            image_data = np_im_to_data(np_image)
            graph.erase()
            height,w,c = np_image.shape
            graph.draw_image(data=image_data, location=(0, height))


        if event =="Resize":
            np_image = open_window_resize(np_image)

            imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)
            image_data = np_im_to_data(np_image)
            graph.erase()
            height,w,c = np_image.shape
            graph.draw_image(data=image_data, location=(0, height))

        if event == 'Save':
            filename = open_window_save(filename,np_image)

        if event == 'Cut':
            np_image = open_window_cut(np_image)
            imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)
            image_data = np_im_to_data(np_image)
            graph.erase()
            height,w,c = np_image.shape
            graph.draw_image(data=image_data, location=(0, height))


        if event == 'Alphabet':
            word = values['-WORD-']
            wordcolour = values['-WORDCOLOUR-']
            if wordcolour =='black':
                wordcolour = [0,0,0]
            elif wordcolour =='white':
                wordcolour = [255,255,255]
            elif wordcolour =='grey':
                wordcolour = [128,128,128]
            elif wordcolour =='red':
                wordcolour = [255,0,0]
            elif wordcolour =='green':
                wordcolour = [0,128,0]

            elif wordcolour =='blue':
                wordcolour = [0,0,255]
            elif wordcolour =='yellow':
                wordcolour = [255,0,0]
            np_image = alphabet(np_image,word,wordcolour,start_data,end_data)

            imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)

            image_data = np_im_to_data(np_image)

            graph.draw_image(data=image_data, location=(0, height))


    window.close()







def update_image_list(np_image,imagelist,currentimage):
    if len(imagelist)-1>currentimage:
        currentimage+=1
        imagelist[currentimage] = np_image
        del imagelist[currentimage+1:]
    else:
        imagelist.append(np_image)
        currentimage+=1
    return imagelist,currentimage

def opening_display():


    col_1 =[
        [sg.Graph(
            canvas_size=(500, 500),
            graph_bottom_left=(0, 0),
            graph_top_right=(700, 700),
            key='-IMAGE-',
            background_color='white',
            change_submits=True,
            drag_submits=True)],
        [sg.Input('Blank Image',enable_events=True, key='-FILEPATH-'),sg.FileBrowse(key='-IN-',file_types=[("JPEG Files","*.jpeg")]), sg.Button(button_text="Open" ,key="OPEN2"),sg.Button('Quit') ],
        ]



    # ----- Full layout -----
    layout = [
        [sg.Column(col_1)

         ]
    ]


    # Create the window
    window = sg.Window('Photoshop', layout, finalize=True)


    graph = window["-IMAGE-"]

    # graph.draw_image(data=image_data, location=(0, height))



    # Event loop
    while True:

        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Quit':
            break


        elif event == "OPEN2":
            window.close()
            display_image(np_image,filename)

            window.close()

        else:
            if values['-FILEPATH-'].endswith('.jpeg'):
                np_image = cv2.imread(values['-FILEPATH-'])
                np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
                filename = values['-FILEPATH-'].split("/")[-1]
                image_data = np_im_to_data(np_image)

                if len(np_image.shape) ==2:
                    h,w = np_image.shape
                else:
                    h,w,c = np_image.shape

                height = h
                width = w

                graph.draw_image(data=image_data, location=(0, height))



    window.close()

def main():
    # parser = argparse.ArgumentParser(description='A simple image viewer.')
    # parser.add_argument('file', action='store', help='Image file.')
    # args = parser.parse_args()
    #
    # print(f'Loading {args.file} ... ', end='')
    # image = cv2.imread(args.file)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(f'{image.shape}')

    # display_image(image, args.file)

    opening_display()

if __name__ == '__main__':
    main()
