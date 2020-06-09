import numpy as np


def circle_transform(z, max_val):
    cos_val = np.cos(2*np.pi*z/max_val)
    sin_val = np.sin(2*np.pi*z/max_val)
    return cos_val, sin_val
