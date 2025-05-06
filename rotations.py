import numpy as np
import cv2

def compute_alignment_angle(cp, np_point):
    delta_y = np_point[1] - cp[1]
    delta_x = np_point[0] - cp[0]
    theta = np.arctan2(delta_y, delta_x) * (180 / np.pi)
    if theta > 90:
        theta -= 180
    elif theta < -90:
        theta += 180
    return theta

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