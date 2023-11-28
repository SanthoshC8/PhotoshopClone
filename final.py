import argparse
import PySimpleGUI as sg
import scipy as sp
from PIL import Image
import PIL
from io import BytesIO
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import colors

import matplotlib.pyplot as plt
import matplotlib
import math
import time
import yaml

matplotlib.use('TkAgg')

#THINGS NEEDED TO BE DONE
# - CHANGE FIRST RESIZE SO IT FITS PROPERLY
# - CHNAGE DESIGN
# - SORT IF STATEMNTS PROPERLY
# - MAKE FUNCTIONS IN DIFFERENT FILES
# - blur incorrect / add redo and fix the grey border
# - fix second image and add image to right
# - comment functions and if statements
# - make the undo/redo into a function
# - add crop
# - fix resize 




def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data



def resize_image(image):
    h,w,c = image.shape

    temp = 480/  w

    neww = 480

    newh = round(h * temp)

    print(newh,neww)

    print(f'Resizing the image to {newh}x{neww} ...', end='')
    image = cv2.resize(image, (neww,newh), interpolation=cv2.INTER_LINEAR)

    return image
    print(f'{image.shape}')

def display_image(np_image,filename):

    # Convert numpy array to data that sg.Graph can understand

    imagelist=[np_image]
    currentimage = 0

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
        sg.Button('Resize'),
        sg.Button('Save'),
        sg.Button('Exit')],

        ]

    row = []


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

    # Event loop
    while True:

        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        if event == "-IMAGE-":  # if there's a "Graph" event, then it's a mouse

            x, y = values["-IMAGE-"]
            #print(x,y)
            if not dragging:
                start_point = [x,y]
                dragging = True
            else:
                end_point = [x,y]
            if prior_rect:
                graph.delete_figure(prior_rect)
            if None not in (start_point, end_point):
                prior_rect = graph.draw_rectangle(start_point, end_point, line_color='blue')

        elif event.endswith('+UP'):  # The drawing has ended because mouse up
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


        if event.endswith("_colour"):

            colour = [int(x) for x in event[:-7].split(",")]


            np_image = changecolour(np_image,colour,start_data,end_data)

            if len(imagelist)-1>currentimage:
                currentimage+=1
                imagelist[currentimage] = np_image
                del imagelist[currentimage+1:]


            else:
                imagelist.append(np_image)
                currentimage+=1


            print("cur",currentimage)
            image_data = np_im_to_data(np_image)
            graph.draw_image(data=image_data, location=(0, height))
            # start_point = [None,None]
            # end_point = [None,None]  # enable grabbing a new rect
            # dragging = False

        if event == "To Gray":
            np_image = togray(np_image,start_data,end_data)

            if len(imagelist)-1>currentimage:
                currentimage+=1
                imagelist[currentimage] = np_image
                del imagelist[currentimage+1:]
            else:
                imagelist.append(np_image)
                currentimage+=1

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

            if len(imagelist)-1>currentimage:
                currentimage+=1
                imagelist[currentimage] = np_image
                del imagelist[currentimage+1:]
            else:
                imagelist.append(np_image)
                currentimage+=1

            print("cur",currentimage)
            image_data = np_im_to_data(np_image)
            graph.erase()
            height,w,c = np_image.shape
            graph.draw_image(data=image_data, location=(0, height))


        if event == 'Averaging':

            np_image =apply_filter_to_image(np_image,[int(values['-SL-']),'a'])

            if len(imagelist)-1>currentimage:
                currentimage+=1
                imagelist[currentimage] = np_image
                del imagelist[currentimage+1:]


            else:
                imagelist.append(np_image)
                currentimage+=1

            image_data = np_im_to_data(np_image)


            graph.draw_image(data=image_data, location=(0, height))



        if event == 'Gaussian':
            tt = gaussian2d(math.sqrt(int(values['-SL-'])*(1/3)), 2*int(values['-SL-'])+1)

            np_image =apply_filter_to_image(np_image,[int(values['-SL-']),tt])

            if len(imagelist)-1>currentimage:
                currentimage+=1
                imagelist[currentimage] = np_image
                del imagelist[currentimage+1:]


            else:
                imagelist.append(np_image)
                currentimage+=1

            image_data = np_im_to_data(np_image)


            graph.draw_image(data=image_data, location=(0, height))




        if event =="Addimage":
            temp = np_image.copy()
            np_image = doubletheimage(np_image,temp)


            if len(imagelist)-1>currentimage:
                currentimage+=1
                imagelist[currentimage] = np_image
                del imagelist[currentimage+1:]
            else:
                imagelist.append(np_image)
                currentimage+=1

            image_data = np_im_to_data(np_image)
            graph.erase()
            height,w,c = np_image.shape
            graph.draw_image(data=image_data, location=(0, height))





        if event =="Resize":
            np_image = open_window_resize(np_image)

            image_data = np_im_to_data(np_image)
            graph.erase()
            height,w,c = np_image.shape
            graph.draw_image(data=image_data, location=(0, height))

        if event == 'Save':
            filename = open_window_save(filename,np_image)



    window.close()


def doubletheimage(image,image2):
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
            res = image.copy()
            res = np.r_[res,newimage    ]
            window.close()
            return res
            break

    window.close()



    # res = image.copy()
    #
    # res = np.r_[res,image2    ]
    #
    #
    # return res













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

        #print(h)
        #print("_________________________")

        return round(np.dot(np.reshape(patch, (hxw*hxw)), np.reshape(h, (hxw*hxw))))

    else:
        hxw = 2*filter[0] + 1

        #h = gaussian2d(7, hxw)

        h = filter[1]
        #print(h)
        #print(h)
        #print("_________________________")
        #print(round((np.dot(np.reshape(patch, (hxw*hxw)), np.reshape(h, (hxw*hxw))))/np.sum(h)))
        #return round(np.dot(np.reshape(patch, (hxw*hxw)), np.reshape(h, (hxw*hxw))))
        return round((np.dot(np.reshape(patch, (hxw*hxw)), np.reshape(h, (hxw*hxw))))/np.sum(h))



def apply_filter_to_image(image, filter):
    #I_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    I_red = image[:,:,0].copy()
    I_green = image[:,:,1].copy()
    I_blue = image[:,:,2].copy()

    imagetest = image.copy()
    h,w ,c= image.shape

    halfw = filter[0]
    fullw = 2*halfw +1


    for y in range(h-fullw+1):
        midy = y + halfw
        for i in range(w-fullw+1):
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
    imagetest[0:halfw,:-1] = 100
    imagetest[:-1,-halfw:] = 100
    imagetest[-halfw:,:0:-1] = 100
    imagetest[::-1,0:halfw] = 100

    return imagetest


def open_window_resize(np_image):

    layout = [[sg.Text("height"),sg.Input('',enable_events=True, key='-HEIGHTINPUT-'),sg.Text("px")],
    [sg.Text("width"),sg.Input('',enable_events=True, key='-WIDTHINPUT-'),sg.Text("px")],
    [sg.Text("constrained"),sg.Checkbox('', key='s1')],
    [sg.Button('NN'),sg.Button('Bilinear'),sg.Button('Cancel')]
    ]

    window = sg.Window("Save", layout, modal=True)

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
            new_image[i][x] = image[round(curr_h)][round(curr_w)]
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
            np_image = Image.fromarray(np_image)
            np_image = np_image.save(temp)

            window.close()
            return temp
            break


    window.close()

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



    print(cropImage.shape)
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




def main():
    parser = argparse.ArgumentParser(description='A simple image viewer.')

    parser.add_argument('file', action='store', help='Image file.')
    args = parser.parse_args()


    print(f'Loading {args.file} ... ', end='')
    image = cv2.imread(args.file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f'{image.shape}')


    image = resize_image(image)

    display_image(image,args.file)






if __name__ == '__main__':
    main()
