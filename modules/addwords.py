import cv2
from modules.crop import *
from modules.resize import *

def alphabet(image,word,wordcolour,start,end):
    height ,width ,c = image.shape
    startheight = max(start[1],end[1])
    startheight = height - startheight
    endheight = min(start[1],end[1])
    endheight = height - endheight
    startwidth = min(start[0],end[0])
    endwidth = max(start[0],end[0])

    alphImage = image.copy()


    n,p=(endwidth-startwidth),len(word)
    c,r=divmod(n,p)
    wsize=[c]*(p-r) + [c+1]*r
    lettercount = 0
    for letter in word:
        if lettercount==0:
            if 65<=ord(letter) <=90 or 97<=ord(letter) <=122 :
                imageword = cv2.imread('letters/{}.png'.format(letter.upper()))
            else:
                imageword = cv2.imread('letters/blank.png')
            imageword = cv2.cvtColor(imageword, cv2.COLOR_BGR2RGB)
            h2,w2,c2 = imageword.shape
            imageword = tocrop(imageword,[0,0+53],[w2,h2])
            imageword = tocrop(imageword,[0,0],[w2,w2-24])
            imageword = resizebil(imageword,endheight-startheight , wsize[lettercount])
        else:
            if 65<=ord(letter) <=90 or 97<=ord(letter) <=122 :
                imageletter = cv2.imread('letters/{}.png'.format(letter.upper()))
            else:
                imageletter = cv2.imread('letters/blank.png')
            imageletter = cv2.cvtColor(imageletter, cv2.COLOR_BGR2RGB)
            h2,w2,c2 = imageletter.shape
            imageletter = tocrop(imageletter,[0,0+53],[w2,h2])
            imageletter = tocrop(imageletter,[0,0],[w2,w2-24])
            imageletter = resizebil(imageletter,endheight-startheight ,wsize[lettercount])
            imageword = np.hstack([imageword,imageletter    ] )

        lettercount+=1


    y2=0
    for y in range(startheight,endheight):
        x2=0
        for x in range(startwidth,endwidth):


            if (imageword[y2][x2] == [0, 0, 0]).all():

                alphImage[y][x] = wordcolour


            x2+=1
        y2+=1


    return alphImage
