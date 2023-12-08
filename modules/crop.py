import numpy as np
def tocrop(image,start,end):
    height ,width ,c = image.shape

    startheight = max(start[1],end[1])
    startheight = height - startheight
    endheight = min(start[1],end[1])
    endheight = height - endheight
    startwidth = min(start[0],end[0])
    endwidth = max(start[0],end[0])

    cropImage = np.zeros((endheight-startheight,endwidth-startwidth,3))

    index_y=0
    index_x =0

    for y in range(startheight,endheight):
        index_x =0
        for x in range(startwidth,endwidth):
            cropImage[index_y][index_x] = image[y][x]
            index_x+=1

        index_y+=1

    return cropImage
