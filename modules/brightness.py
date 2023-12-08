import cv2
def change_brightness(image,value,start,end):


    hsv = image.copy()

    height ,width ,c = image.shape
    startheight = max(start[1],end[1])
    startheight = height - startheight
    endheight = min(start[1],end[1])
    endheight = height - endheight
    startwidth = min(start[0],end[0])
    endwidth = max(start[0],end[0])


    hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2HSV)


    lim = 255 - value


    for y in range(startheight,endheight):
        for x in range(startwidth,endwidth):
            if hsv[y][x][2]+value <0:
                hsv[y][x][2] = 0
            elif hsv[y][x][2] >lim:
                hsv[y][x][2] = 255

            elif hsv[y][x][2] <=lim:
                hsv[y][x][2] +=value




    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img
