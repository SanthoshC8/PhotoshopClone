def changecolour(image,colour,start,end):

    colourImage = image.copy()

    if start==end:
        colourImage[start[0]][start[1]] = colour
        return colourImage


    height ,width ,c = image.shape
    startheight = max(start[1],end[1])
    startheight = height - startheight
    endheight = min(start[1],end[1])
    endheight = height - endheight
    startwidth = min(start[0],end[0])
    endwidth = max(start[0],end[0])



    for y in range(startheight,endheight):
        for x in range(startwidth,endwidth):
            colourImage[y][x] = colour

    return colourImage
