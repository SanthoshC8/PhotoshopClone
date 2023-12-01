import argparse
import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from resize import *

import time

matplotlib.use('TkAgg')

#THINGS NEEDED TO BE DONE
# - CHNAGE DESIGN
# - SORT IF STATEMNTS PROPERLY
# - MAKE FUNCTIONS IN DIFFERENT FILES under modules folder
# - comment functions and if statements
# - fix spaces between functions
# - add default to combo
# - add opening page for photoshop
# - delete print statemnts




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


    # Define the layout
    layout = [[sg.Graph(
        canvas_size=(700, 700),
        graph_bottom_left=(0, 0),
        graph_top_right=(700, 700),
        key='-IMAGE-',
        background_color='white',
        change_submits=True,
        drag_submits=True),
        sg.Button('Change Image'),
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
        sg.Text(key='info', size=(60, 1)),
        sg.Button('Gray',key="To Gray")],
        [sg.Button('undo',key="Undo"),
        sg.Button('redo',key="Redo"),
        sg.Button('crop',key="Crop"),
        sg.Slider(range=(0,15),default_value=0,expand_x=True, enable_events=True,
        orientation='horizontal', key='-SL-'),
        sg.Button('Averaging'),
        sg.Button('Gaussian'),
        sg.Button('Addimage'),
        sg.Combo(['up','down','left','right','custom'] ,size=(20, 4),readonly=True, enable_events=True, key='-RESIZEOP-'),
        sg.Button('Resize'),
        sg.Button('Save'),
        sg.Button('Exit')],
        [sg.Slider(range=(0,15),default_value=0,expand_x=True, enable_events=True,
        orientation='horizontal', key='-MOUSESIZE-'),
        sg.Combo(['brush','sel',] ,size=(20, 4),readonly=True, enable_events=True, key='-MOUSE-'),
        sg.Button('Cut'),
        ]

        ]


    # Create the window
    window = sg.Window('Display Image', layout, finalize=True)


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
            if values['-MOUSE-'] != 'brush':

                if not dragging:
                    start_point = [x,y]
                    dragging = True
                else:
                    end_point = [x,y]
                if prior_rect:
                    graph.delete_figure(prior_rect)
                if None not in (start_point, end_point):
                    prior_rect = graph.draw_rectangle(start_point, end_point, line_color='blue')
            else:
                start_time = time.time()

                start_data = [x,y]
                end_data = [x+int(values["-MOUSESIZE-"]),y+int(values["-MOUSESIZE-"])]
                if brush_choice[0] != "":
                    if brush_choice[0] == "_colour":
                        np_image = changecolour(np_image,brush_choice[1],start_data,end_data)

                    elif brush_choice[0] == "To Gray":
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
                    # print(round(end_time-start_time,2))








        elif event.endswith('+UP'):  # The drawing has ended because mouse up
            if values['-MOUSE-'] != 'brush':
                info = window["info"]
                start_data = start_point
                end_data = end_point
                info.update(value=f"grabbed rectangle from {start_data[0],start_data[1]} to {end_data[0],end_data[1]}")

                #np_image = pixeltoblack(np_image,start_point,end_point)
                #image_data = np_im_to_data(np_image)
                #graph.draw_image(data=image_data, location=(0, height))

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

            else:
                np_image = changecolour(np_image,colour,start_data,end_data)

                imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)


                print("cur",currentimage)
                image_data = np_im_to_data(np_image)
                graph.draw_image(data=image_data, location=(0, height))
                # start_point = [None,None]
                # end_point = [None,None]  # enable grabbing a new rect
                # dragging = False


        if event == "To Gray":

            if values['-MOUSE-'] == 'brush':
                brush_choice = ['To Gray',None]

            else:
                np_image = togray(np_image,start_data,end_data)

                imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)

                print("cur",currentimage)
                image_data = np_im_to_data(np_image)
                graph.draw_image(data=image_data, location=(0, height))


        if event =="Undo":
            if currentimage>0:
                currentimage-=1
                print("cur",currentimage)
                np_image=imagelist[currentimage]
                image_data = np_im_to_data(np_image)
                graph.erase()
                height,w,c = np_image.shape
                graph.draw_image(data=image_data, location=(0, height))


        if event =="Redo":
            if len(imagelist)-1>currentimage:
                currentimage+=1
                print("cur",currentimage)
                np_image=imagelist[currentimage]
                image_data = np_im_to_data(np_image)
                graph.erase()
                height,w,c = np_image.shape
                graph.draw_image(data=image_data, location=(0, height))

        if event =="Crop":
            np_image = tocrop(np_image,start_data,end_data)

            imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)

            print("cur",currentimage)
            image_data = np_im_to_data(np_image)
            graph.erase()
            height,w,c = np_image.shape
            graph.draw_image(data=image_data, location=(0, height))


        if event == 'Averaging':
            if values['-MOUSE-'] == 'brush':
                brush_choice = ['Averaging',[int(values['-SL-']),'a']]

            else:

                np_image =apply_filter_to_image(np_image,[int(values['-SL-']),'a'],start_data,end_data)

                imagelist,currentimage=update_image_list(np_image,imagelist,currentimage)

                image_data = np_im_to_data(np_image)


                graph.draw_image(data=image_data, location=(0, height))


        if event == 'Gaussian':
            tt = gaussian2d(math.sqrt(int(values['-SL-'])*(1/3)), 2*int(values['-SL-'])+1)

            if values['-MOUSE-'] == 'brush':
                brush_choice = ['Gaussian',[int(values['-SL-']),tt]]

            else:

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



    window.close()


def open_window_cut(np_image):
    px = 0
    layout = [[sg.Text("cut"),
    sg.Input(px,enable_events=True, key='-INPUT-'),
    sg.Text("px from the"),
    sg.Combo(['top','bottom','left','right'] ,size=(20, 4),readonly=True, enable_events=True, key='-ORDER-'),


    ],
    [sg.Button('Cut'),sg.Button('Cancel')]
    ]

    window = sg.Window("Save", layout, modal=True)

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


def update_image_list(np_image,imagelist,currentimage):
    if len(imagelist)-1>currentimage:
        currentimage+=1
        imagelist[currentimage] = np_image
        del imagelist[currentimage+1:]
    else:
        imagelist.append(np_image)
        currentimage+=1
    return imagelist,currentimage


def addimage(image,order,start,end):
    image_h,image_w,c = image.shape
    layout = [[sg.Text("file:"),sg.Input('',enable_events=True, key='-FILEPATH-'),sg.FileBrowse(key='-IN-',file_types=[("JPEG Files","*.jpeg")])],
    [sg.Button('Submit'),sg.Button('Cancel')]
    ]

    window = sg.Window("Add Image", layout, modal=True)
    while True:
        event, values = window.read()

        if event == 'Cancel' or event == sg.WIN_CLOSED:
            window.close()
            return image
            break

        if event == 'Submit':

            newimage = cv2.imread(values["-FILEPATH-"])
            newimage = cv2.cvtColor(newimage, cv2.COLOR_BGR2RGB)

            #newimage_height,newimage_width,newimage_c = newimage.shape

            #newimage_height , newimage_width = constrained(newimage,image_h,image_w)
            # np_image = resizebil(newimage,newimage_height , newimage_width)

            #newimage = resize_image(newimage,316 ,480)

            res = image.copy()

            if order=='up':
                print('-------------------')
                print(image_h,image_w)
                newimage_height , newimage_width = constrained(newimage,None,image_w)
                print(newimage_height,newimage_width)

                newimage = resizebil(newimage,newimage_height , newimage_width)
                res = np.vstack([newimage,res    ] )
            elif order=='down':
                newimage_height , newimage_width = constrained(newimage,None,image_w)
                newimage = resizebil(newimage,newimage_height , newimage_width)
                res = np.vstack([res,newimage    ] )

            elif order=='left':
                newimage_height , newimage_width = constrained(newimage,image_h,None)
                newimage = resizebil(newimage,newimage_height , newimage_width)
                res = np.hstack([newimage,res    ] )

            elif order=='right':
                newimage_height , newimage_width = constrained(newimage,image_h,None)
                newimage = resizebil(newimage,newimage_height , newimage_width)
                res = np.hstack([res,newimage    ] )

            else:
                print('cusom')



                height ,width ,c = image.shape

                startheight = max(start[1],end[1])

                startheight = height - startheight

                endheight = min(start[1],end[1])

                endheight = height - endheight

                startwidth = min(start[0],end[0])

                endwidth = max(start[0],end[0])


                newimage = resizebil(newimage,endheight-startheight , endwidth-startwidth)



                newimage_y=0
                for y in range(startheight,endheight):
                    newimage_x=0
                    for i in range(startwidth,endwidth):
                        res[y][i] = newimage[newimage_y][newimage_x]
                        newimage_x+=1
                    newimage_y+=1





            #res = np.r_[res,newimage    ]  #adds to bottom must be 480

            #res = np.vstack([res,newimage    ] )

            #res = np.hstack([res,newimage    ] ) #adds to right must be 316 (316,480)
            window.close()
            return res
            break

    window.close()


def gaussian2_xy(mean, cov, xy):
    invcov = np.linalg.inv(cov)
    results = np.ones([xy.shape[0], xy.shape[1]])
    for x in range(0, xy.shape[0]):
        for y in range(0, xy.shape[1]):
            v = xy[x,y,:].reshape(2,1) - mean
            results[x,y] = np.dot(np.dot(np.transpose(v), invcov), v)
    results = np.exp( - results / 2 )
    return results


def gaussian2_n(mean, cov, n):
    s = n//2
    x = np.linspace(-s,s,n)
    y = np.linspace(-s,s,n)
    xc, yc = np.meshgrid(x, y)
    xy = np.zeros([n, n, 2])
    xy[:,:,0] = xc
    xy[:,:,1] = yc

    return gaussian2_xy(mean, cov, xy), xc, yc


def gaussian2d(var, n):
    mean =  np.array([0, 0])
    mean = mean.reshape(2,1)
    cov = np.array([[var,0],[0,var]])
    k, xc, yc = gaussian2_n(mean, cov, n)
    return k


def apply_filter_to_patch(patch,filter):


    if filter[1] =="a":
        hxw = 2*filter[0] + 1
        h = (np.ones(hxw*hxw).reshape(hxw,hxw) )/ (hxw*hxw)
        return round(np.dot(np.reshape(patch, (hxw*hxw)), np.reshape(h, (hxw*hxw))))

    else:
        hxw = 2*filter[0] + 1
        h = filter[1]
        return round((np.dot(np.reshape(patch, (hxw*hxw)), np.reshape(h, (hxw*hxw))))/np.sum(h))


def apply_filter_to_image(image, filter,start,end):

    height ,width ,c = image.shape
    startheight = max(start[1],end[1])
    startheight = height - startheight
    endheight = min(start[1],end[1])
    endheight = height - endheight
    startwidth = min(start[0],end[0])
    endwidth = max(start[0],end[0])

    I_red = image[:,:,0].copy()
    I_green = image[:,:,1].copy()
    I_blue = image[:,:,2].copy()

    imagetest = image.copy()
    h,w ,c= image.shape

    halfw = filter[0]
    fullw = 2*halfw +1


    for y in range(startheight,endheight):
        midy = y + halfw
        for i in range(startwidth,endwidth):
            midi = i + halfw

            temppatch = I_red[y:y+fullw,i:i+fullw]
            newnum = apply_filter_to_patch(temppatch,filter)

            temppatch2 = I_green[y:y+fullw,i:i+fullw]
            newnum2 = apply_filter_to_patch(temppatch2,filter)

            temppatch3 = I_blue[y:y+fullw,i:i+fullw]
            newnum3 = apply_filter_to_patch(temppatch3,filter)

            imagetest[midy,midi] = [newnum,newnum2,newnum3]
            #print(temppatch)

    #boundary
    # imagetest[0:halfw,:-1] = 100
    # imagetest[:-1,-halfw:] = 100
    # imagetest[-halfw:,:0:-1] = 100
    # imagetest[::-1,0:halfw] = 100

    return imagetest


def tocrop(image,start,end):
    height ,width ,c = image.shape

    startheight = max(start[1],end[1])

    startheight = height - startheight

    endheight = min(start[1],end[1])

    endheight = height - endheight

    startwidth = min(start[0],end[0])

    endwidth = max(start[0],end[0])
    print(startheight-endheight,endwidth-startwidth)

    cropImage = np.zeros((endheight-startheight,endwidth-startwidth,3))

    #cropImage = image.copy()

    index_y=0
    index_x =0

    for y in range(startheight,endheight):
        index_x =0

        for x in range(startwidth,endwidth):
            print(index_x)

            cropImage[index_y][index_x] = image[y][x]

            index_x+=1

        index_y+=1







    return cropImage


def changecolour(image,colour,start,end):

    height ,width ,c = image.shape


    startheight = max(start[1],end[1])
    startheight = height - startheight
    endheight = min(start[1],end[1])
    endheight = height - endheight
    startwidth = min(start[0],end[0])
    endwidth = max(start[0],end[0])

    colourImage = image.copy()

    for y in range(startheight,endheight):
        for x in range(startwidth,endwidth):
            colourImage[y][x] = colour



    return colourImage


def togray(image,start,end, color_key=None):


    height ,width ,c = image.shape

    startheight = max(start[1],end[1])
    startheight = height - startheight
    endheight = min(start[1],end[1])
    endheight = height - endheight
    startwidth = min(start[0],end[0])
    endwidth = max(start[0],end[0])

    #image[startheight:endheight, startwidth:endwidth, 0] = 255

    #grayImage = np.zeros(image.shape)
    R = np.array(image[startheight:endheight, startwidth:endwidth, 0])
    G = np.array(image[startheight:endheight, startwidth:endwidth, 1])
    B = np.array(image[startheight:endheight, startwidth:endwidth, 2])


    R = (R *.299)
    G = (G *.587)
    B = (B *.114)

    Avg = (R+G+B)
    grayImage = image.copy()

    for i in range(3):
       grayImage[startheight:endheight, startwidth:endwidth,i] = Avg

    return grayImage


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

            #np_image = Image.fromarray(np_image)
            np_image = np_image.save(temp)
            window.close()
            return temp
            break

    window.close()


def main():
    parser = argparse.ArgumentParser(description='A simple image viewer.')
    parser.add_argument('file', action='store', help='Image file.')
    args = parser.parse_args()

    print(f'Loading {args.file} ... ', end='')
    image = cv2.imread(args.file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f'{image.shape}')

    display_image(image,args.file)


if __name__ == '__main__':
    main()
