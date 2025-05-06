import numpy as np

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