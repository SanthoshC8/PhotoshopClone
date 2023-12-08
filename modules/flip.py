import cv2
import numpy as np
def toflip(image,dir,start,end):

    flippedimage = image.copy()

    height ,width ,c = image.shape
    startheight = max(start[1],end[1])
    startheight = height - startheight
    endheight = min(start[1],end[1])
    endheight = height - endheight
    startwidth = min(start[0],end[0])
    endwidth = max(start[0],end[0])


    if dir == 'custom vertically':

        flippedimage[startheight:endheight,startwidth:endwidth:] = flippedimage[endheight:startheight:-1,startwidth:endwidth]

    elif dir == 'custom horizontally':
        flippedimage[startheight:endheight,startwidth:endwidth:] = flippedimage[startheight:endheight,endwidth:startwidth:-1]


    elif dir == 'vertically':
        flippedimage= np.flipud(flippedimage)
        # flippedimage = cv2.flip(flippedimage, 0)
    else:
        flippedimage= np.fliplr(flippedimage)
        # flippedimage = cv2.flip(flippedimage, 1)


    return flippedimage
