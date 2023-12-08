import numpy as np
def torotate(image,dir,start,end):

    rotimage = image.copy()


    if dir == '90 deg':
        rotimage= np.rot90(rotimage,3)
        # flippedimage = cv2.flip(flippedimage, 0)
    elif dir == '180 deg':
        rotimage= np.rot90(rotimage,2)
        # flippedimage = cv2.flip(flippedimage, 1)

    elif dir == '270 deg':
        rotimage= np.rot90(rotimage)



    return rotimage
