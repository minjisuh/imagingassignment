import numpy as np
from matplotlib import pyplot as plt

def das_beamforming(rf_data, speed_of_sound, fs, spacing, element_pos):
    # rf_data: RF data
    # speed_of_sound: speed of sound
    # fs: sampling frequency
    # spacing: distance between elements
    # element_pos: position of elements

    num_sensors, num_samples = rf_data.shape
    num_points = element_pos.shape[0]
    image = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            x = i * spacing
            y = j * spacing
            for sensor in range(num_sensors):
                distance = np.sqrt((element_pos[sensor, 0] - x) ** 2 + (element_pos[sensor, 1] - y) ** 2)
                time_delay = distance / speed_of_sound
                sample_delay = int(time_delay * fs)
                if sample_delay < num_samples:
                    image[i, j] += rf_data[sensor, sample_delay]
    return image


def log_compression(image):
    return np.log(1 + np.abs(image))

rf_data = np.load('arm_rfdata.npy')
rf_data_frame = rf_data[:, :, 0]
speed_of_sound = 1540
fs = 7.6e6 * 4.1
spacing = 0.0003
# element_pos 배열 생성


compressed_image = log_compression(das_beamforming(rf_data_frame, speed_of_sound, fs, spacing, element_pos))

plt.imshow(compressed_image, cmap='gray')
plt.show()