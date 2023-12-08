import PySimpleGUI as sg
import cv2
from modules.resize import *

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
            res = image.copy()

            if order=='top':
                newimage_height , newimage_width = constrained(newimage,None,image_w)

                newimage = resizebil(newimage,newimage_height , newimage_width)
                res = np.vstack([newimage,res    ] )
            elif order=='bottom':
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

            window.close()
            return res
            break
    window.close()
