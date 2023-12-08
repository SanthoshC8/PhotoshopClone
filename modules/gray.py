import numpy as np

def togray(image,start,end, color_key=None):

    height ,width ,c = image.shape

    grayImage = image.copy()

    if start==end:
        R = np.array(image[start[0]:start[0]+1, start[1]:start[1]+1, 0])
        G = np.array(image[start[0]:start[0]+1, start[1]:start[1]+1, 1])
        B = np.array(image[start[0]:start[0]+1, start[1]:start[1]+1, 2])

        R = (R *.299)
        G = (G *.587)
        B = (B *.114)
        Avg = (R+G+B)
        # print(Avg[0][0])
        # for i in range(3):
        #    grayImage[start[0]:start[0]+1, start[1]:start[1]+1,i] = Avg
        y= start[0]
        x = start[1]
        grayImage[y][x][0] = Avg[0][0]
        grayImage[start[0], start[1],1] = Avg[0][0]
        grayImage[start[0], start[1],2] = Avg[0][0]
        return grayImage


    else:
        startheight = max(start[1],end[1])
        startheight = height - startheight
        endheight = min(start[1],end[1])
        endheight = height - endheight
        startwidth = min(start[0],end[0])
        endwidth = max(start[0],end[0])

        R = np.array(image[startheight:endheight, startwidth:endwidth, 0])
        G = np.array(image[startheight:endheight, startwidth:endwidth, 1])
        B = np.array(image[startheight:endheight, startwidth:endwidth, 2])

        R = (R *.299)
        G = (G *.587)
        B = (B *.114)
        Avg = (R+G+B)




        for i in range(3):
           grayImage[startheight:endheight, startwidth:endwidth,i] = Avg

        return grayImage
