import cv2
def change_saturation(image,value,start,end):


    sat = image.copy()

    height ,width ,c = image.shape
    startheight = max(start[1],end[1])
    startheight = height - startheight
    endheight = min(start[1],end[1])
    endheight = height - endheight
    startwidth = min(start[0],end[0])
    endwidth = max(start[0],end[0])


    sat = cv2.cvtColor(sat, cv2.COLOR_RGB2HSV)


    lim = 255 - value


    for y in range(startheight,endheight):
        for x in range(startwidth,endwidth):
            if sat[y][x][1]+value <0:
                sat[y][x][1] = 0
            elif sat[y][x][1] >lim:
                sat[y][x][1] = 255

            elif sat[y][x][1] <=lim:
                sat[y][x][1] +=value




    img = cv2.cvtColor(sat, cv2.COLOR_HSV2RGB)
    return img
