import numpy as np
import cv2 as cv

def compute_cp(image):
    rows, cols = image.shape
    total_intensity = np.sum(image)

    if total_intensity == 0:
        return (cols // 2, rows // 2) 

    x_coords = np.tile(np.arange(cols), (rows, 1))
    y_coords = np.tile(np.arange(rows).reshape(rows, 1), (1, cols))

    x_c = np.sum(x_coords * image) / total_intensity
    y_c = np.sum(y_coords * image) / total_intensity

    return int(x_c), int(y_c)

def get_cp(img):
    #https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
    ret, thresh = cv.threshold(img, 5,255, 0)
    M = cv.moments(thresh)

    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    return cX, cY

def cp_all(stack):
    output = np.zeros((stack.shape[0],2))
    for i, img in enumerate(stack):
        cp = compute_cp(img)
        output[i] = cp

    return output

def compute_np_left(image, intensity_thresh=20):
    rows, cols = image.shape
    lower_start = int(rows * 0.85) 
    bottom = image[lower_start:, :]

    for x in range(0, int(cols * 0.5)):
        col = bottom[:, x]
        if np.any(col > intensity_thresh):
            y_vals = np.where(col > intensity_thresh)[0]
            if len(y_vals) > 0:
                y_max = lower_start + int(np.max(y_vals)) 
                return x, y_max

    return 0, int(rows * 0.75)

def get_np(img):
    h,w = img.shape
    Y = h//2
    yline = img[Y,:]
    startX = np.min(yline.nonzero())
    endX = np.max(yline.nonzero())
    X = (endX - startX) //2 + startX
    xline = img[:,X]
    startY = np.min(xline.nonzero()) 
    endY = np.max(xline.nonzero()) 
    third = (endY - startY) // 3
    top = int(endY)
    bottom = int((endY - third))
    bottomthird = img[bottom:top, :]
    mins = []
    for i in range(bottomthird.shape[0]):
        mins.append(np.min(bottomthird[i,:].nonzero()))

    minx = min(mins)
    minxline = bottomthird[:,minx]
    maxy = np.max(minxline.nonzero())
  

    return (minx, bottom + maxy)

def np_all(stack):
    output = np.zeros((stack.shape[0],2))
    for i, img in enumerate(stack):
        cp = get_np(img)
        output[i] = cp

    return output