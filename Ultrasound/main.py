import numpy as np
from matplotlib import pyplot as plt


rf_data = np.load('arm_rfdata.npy')

def das_beamforming(rf_data, c, fs, pitch, element_pos, image_size, image_res):
    # rf_data: RF data
    # c: speed of sound
    # fs: sampling frequency
    # pitch: distance between elements
    # element_pos: position of elements
    # image_size: size of image
    # image_res: resolution of image

    return image