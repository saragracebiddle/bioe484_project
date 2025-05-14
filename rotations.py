import numpy as np
import cv2
from find_points import *

def compute_alignment_angle(CP, NP):
    theta = np.arctan((NP[0] - CP[0]) / (NP[1]-CP[1]))
    return theta

def avg_theta(CPS, NPS):
    thetas = np.zeros((CPS.shape[0],1))
    for i, img in enumerate(thetas):
        theta = compute_alignment_angle(CPS[i], NPS[i])
        thetas[i] = theta

    avg_theta = thetas.mean()
    
    return np.rad2deg(avg_theta), np.rad2deg(thetas)


def rotate_image(image, angle, image_center):
  #https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
  #image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def rotate_all(stack):
    nps = np_all(stack)
    cps = cp_all(stack)

    avgtheta, allthetas  = avg_theta(cps, nps)

    output = stack.copy()
    for i, img in enumerate(stack):
        rotate_by = int(avgtheta - allthetas[i])

        if avgtheta > allthetas[i]:
            rotate_by = -rotate_by
        output[i] = rotate_image(img, int(rotate_by), cps[i])

    return output


def rotate_image_about_cp_square(image, cp, angle_deg):
    h, w = image.shape
    side = max(h, w)

    square_canvas = np.zeros((side, side), dtype=image.dtype)

    y_offset = (side - h) // 2
    x_offset = (side - w) // 2
    square_canvas[y_offset:y_offset+h, x_offset:x_offset+w] = image.copy()

    cp_square = (cp[0] + x_offset, cp[1] + y_offset)

    M = cv2.getRotationMatrix2D(center=cp_square, angle=angle_deg, scale=1.0)
    rotated = cv2.warpAffine(square_canvas, M, (side, side), flags=cv2.INTER_LINEAR)

    return rotated, cp_square