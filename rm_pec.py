import numpy as np
import cv2 as cv
from scipy.signal import convolve2d

def remove_pectoral_muscle(image, background_thresh=30, segment_count=5):
    img = image.copy()
    rows, cols = img.shape
    mcontour = []
    max_scan_height = int(rows * 0.9)  

    for y in range(0, max_scan_height, 5):
        x_start = int(cols - ((1 - (y / max_scan_height)) * (cols * 0.52)))
        for x in range(x_start, int(cols * 0.5), -1):
            patch = img[max(y-1, 0):min(y+2, rows), x-1:x+2]
            if np.mean(patch) > background_thresh:
                mcontour.append((x, y))
                break

        for x in range(x_start, 20, -1):
            patch = img[max(y-1, 0):min(y+2, rows), x-1:x+2]
            if np.mean(patch) > background_thresh:
                mcontour.append((x, y))
                break

    if len(mcontour) < 2:
        print("[Warning] Not enough contour points found.")
        return img, np.empty((0, 2))

    y_offset = 150
    contour_pts = np.array(mcontour)
    contour_pts[:, 1] = np.clip(contour_pts[:, 1] - y_offset, 0, rows - 1)

    smoothed = []
    for i in range(segment_count):
        start = int(i * len(contour_pts) / segment_count)
        end = int((i + 1) * len(contour_pts) / segment_count)
        segment = contour_pts[start:end]
        if len(segment) > 0:
            avg_x = int(np.mean(segment[:, 0]))
            avg_y = int(np.mean(segment[:, 1]))
            smoothed.append((avg_x, avg_y))

    smoothed = np.array(smoothed, dtype=np.int32)

    if len(smoothed) > 2:
        smoothed = smoothed[:-1]


    mask = np.ones_like(img, dtype=np.uint8) * 255
    poly = np.array(
        [[cols, 0]] + smoothed.tolist() + [[cols, rows]],
        dtype=np.int32
    )


    cv.fillPoly(mask, [poly], 0)

    cleaned = cv.bitwise_and(img, img, mask=mask)

    ret, threshold = cv.threshold(cleaned, 20, 255, cv.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8) 
    eroded = cv.erode(threshold, kernel, iterations = 1)  
    kernel = np.ones((150, 150), np.uint8) 
    dilated = cv.dilate(eroded, kernel, iterations = 1)  
    output = np.zeros(threshold.shape, dtype=np.uint8)
    mask = cv.bitwise_or(output, dilated)

    tfmask = np.equal(mask, 255)
    np.copyto(output, img, where = tfmask)
    return output, smoothed


def pec_cntrs(img, background_thresh = 30):
    kernel = np.array([[1/9,1/9,1/9],
                   [1/9,1/9,1/9],
                   [1/9,1/9,1/9]])
    conv = convolve2d(img, kernel, mode = 'same')
    ret, thresh = cv.threshold(img, 5,255, cv.THRESH_BINARY)
    cnts, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    c = cnts[0]
    extLeft = tuple(c[c[:,:,0].argmin()][0])
    extRight = tuple(c[c[:,:,0].argmax()][0])
    a = int(extRight[0] - 20)
    b = int(extRight[0])
    rowmeans = img[:, a:b].mean(axis = 1)
    rowmaxes = img[:, a:b].max(axis = 1)
    mcontours = []
    for L1 in range(img.shape[0] -1):
        L2 = L1+1
        rowsmaxL = rowmaxes[[int(L1),int(L2)]].max()
        rowsmeanL = rowmeans[[int(L1),int(L2)]].mean()
        localthresh = 64 + ((rowsmeanL * 0.7 + rowsmaxL * 0.3)/2)
        pts = []
        i = extRight[0] -1 
        val = conv[L1, i]
        if rowsmaxL > background_thresh:
            while val > localthresh:
                pts.append(i)
                i += -1
                val = conv[L1, i]
            else:
                mcontours.append((L1, i))
            
    return mcontours

def rm_cntrs(img, mcontours):
    ys = [x[0] for x in mcontours]
    xs = [x[1] for x in mcontours]

    mid1 = (np.mean(xs[:len(xs)//2]), np.mean(ys[:len(ys)//2]))
    mid2 = (np.mean(xs[len(xs)//2:]), np.mean(ys[len(ys)//2:]))

    img_c = img.copy()
    
    pts = np.array([[xs[1], ys[1]], mid1, mid2, [xs[-1], ys[-1]],[img.shape[1],img.shape[1]], [img.shape[1],0]],  np.int32)
    pts = pts.reshape((-1,1,2))

    poly = cv.fillPoly(img_c, [pts],  0)

    return poly



def rm_pec_all(stack):
    output = stack.copy()
    for i, img in enumerate(stack):
        
        pts = pec_cntrs(img, background_thresh = np.percentile(img, 99))
        if len(pts) > 5:
            output[i] = rm_cntrs(img, pts)
        else:
            output[i] = img

    return output