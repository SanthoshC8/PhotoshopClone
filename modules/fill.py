import scipy as sp
import numpy as np
import cv2
from modules.colour import *
from modules.gray import *
def apply_effects(np_image,start,eff,thresh):

    height, width,c = np_image.shape
    np_image2 = np.copy(np_image)


    gray_image = np.dot(np_image2[..., :3], [0.299, 0.587, 0.114])

    sobel_x = sp.ndimage.sobel(gray_image, axis=0, mode='constant')
    sobel_y = sp.ndimage.sobel(gray_image, axis=1, mode='constant')

    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255

    fill_image = np.copy(np_image)

    startheight = start[1]
    startheight = height - startheight

    startwidth = start[0]

    if eff[0] != None:
        fill_image=floodFill(gradient_magnitude,fill_image,startheight,startwidth,eff,thresh)


    return fill_image


def floodFill( gradient_magnitude,image, sr, sc, eff,thresh):
        height,width,c = image.shape
        tt = gradient_magnitude[sr][sc].copy()

        def dfs(r, c):
            nonlocal height, width,  image
            if r <= 0 or c <= 0 or r>height-1 or c>width-1:
                return
            elif (image[r][c]==eff[1]).all() or gradient_magnitude[r, c]==-666:
                return
            elif not tt-thresh<=gradient_magnitude[r][c]<=tt+thresh:
                return
            if eff[0] == "_colour":
                image = changecolour(image,eff[1],[r,c],[r,c])
            elif eff[0] =="Gray":
                image = togray(image,[r,c],[r,c])
                gradient_magnitude[r, c] = -666
            dfs(r+1,c)
            dfs(r-1,c)
            dfs(r,c+1)
            dfs(r,c-1)
        dfs(sr, sc)
        return image
