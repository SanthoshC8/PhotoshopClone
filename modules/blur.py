import numpy as np

def gaussian2_xy(mean, cov, xy):
    invcov = np.linalg.inv(cov)
    results = np.ones([xy.shape[0], xy.shape[1]])
    for x in range(0, xy.shape[0]):
        for y in range(0, xy.shape[1]):
            v = xy[x,y,:].reshape(2,1) - mean
            results[x,y] = np.dot(np.dot(np.transpose(v), invcov), v)
    results = np.exp( - results / 2 )
    return results


def gaussian2_n(mean, cov, n):
    s = n//2
    x = np.linspace(-s,s,n)
    y = np.linspace(-s,s,n)
    xc, yc = np.meshgrid(x, y)
    xy = np.zeros([n, n, 2])
    xy[:,:,0] = xc
    xy[:,:,1] = yc

    return gaussian2_xy(mean, cov, xy), xc, yc


def gaussian2d(var, n):
    mean =  np.array([0, 0])
    mean = mean.reshape(2,1)
    cov = np.array([[var,0],[0,var]])
    k, xc, yc = gaussian2_n(mean, cov, n)
    return k


def apply_filter_to_patch(patch,filter):


    if filter[1] =="a":
        hxw = 2*filter[0] + 1
        h = (np.ones(hxw*hxw).reshape(hxw,hxw) )/ (hxw*hxw)
        return round(np.dot(np.reshape(patch, (hxw*hxw)), np.reshape(h, (hxw*hxw))))

    else:
        hxw = 2*filter[0] + 1
        h = filter[1]
        return round((np.dot(np.reshape(patch, (hxw*hxw)), np.reshape(h, (hxw*hxw))))/np.sum(h))


def apply_filter_to_image(image, filter,start,end):

    height ,width ,c = image.shape
    startheight = max(start[1],end[1])
    startheight = height - startheight
    endheight = min(start[1],end[1])
    endheight = height - endheight
    startwidth = min(start[0],end[0])
    endwidth = max(start[0],end[0])

    I_red = image[:,:,0].copy()
    I_green = image[:,:,1].copy()
    I_blue = image[:,:,2].copy()

    imagetest = image.copy()
    h,w ,c= image.shape

    halfw = filter[0]
    fullw = 2*halfw +1


    for y in range(startheight,endheight):
        midy = y + halfw
        for i in range(startwidth,endwidth):
            midi = i + halfw

            temppatch = I_red[y:y+fullw,i:i+fullw]
            newnum = apply_filter_to_patch(temppatch,filter)

            temppatch2 = I_green[y:y+fullw,i:i+fullw]
            newnum2 = apply_filter_to_patch(temppatch2,filter)

            temppatch3 = I_blue[y:y+fullw,i:i+fullw]
            newnum3 = apply_filter_to_patch(temppatch3,filter)

            imagetest[midy,midi] = [newnum,newnum2,newnum3]

    #boundary
    # imagetest[0:halfw,:-1] = 100
    # imagetest[:-1,-halfw:] = 100
    # imagetest[-halfw:,:0:-1] = 100
    # imagetest[::-1,0:halfw] = 100

    return imagetest
