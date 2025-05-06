import numpy as np
import cv2

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


    cv2.fillPoly(mask, [poly], 0)

    cleaned = cv2.bitwise_and(img, img, mask=mask)
    return cleaned, smoothed


def rm_pec_all(stack, background_thresh = 10, segment_count = 150):
    output = stack.copy()
    for i, img in enumerate(stack):
        rmd, cutline = remove_pectoral_muscle(img, background_thresh, segment_count)
        output[i] = rmd

    return output