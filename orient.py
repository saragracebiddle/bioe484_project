import numpy as np

def reorient(img):
    return np.fliplr(img)

def reorient_all(stack):
    for i, img in enumerate(stack):
        if i % 2 != 0:
            img = reorient(img)
        
        stack[i] = img
    
    return stack
