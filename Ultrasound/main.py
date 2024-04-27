import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
from scipy.ndimage import zoom
import imageio.v2 as imageio
import os

rf_data = np.load('Ultrasound/arm_rfdata.npy')
phantom_data = np.load('Ultrasound/phantom_rfdata.npy')
center_frequency = 7.6e6  # Hz
speed_of_sound = 1450  # m/s
speed_of_sound_phantom = 1450 # m/s
spacing = 0.3e-3  # m
sampling_rate = 31.16e6  # Hz
sensors = 128  
samples = 2176  
position = np.array([i * spacing for i in range(sensors)])  


# 하나의 프레임에 대한 DAS beamforming
def das_beamforming(rf_data, samples, sensors, position, sampling_rate, speed_of_sound_phantom, frame_number):
    base_das = np.zeros((samples, sensors))
    
    for i in range(samples):
        depth = i * speed_of_sound_phantom / (2 * sampling_rate) 
        for j in range(sensors):
            distance = np.sqrt(depth**2 + (position[j] - position[sensors//2])**2)
            time_delay = distance / speed_of_sound_phantom
            sample_index = int(time_delay *sampling_rate)
            base_das[i, j] += rf_data[sample_index, j, frame_number]
    return base_das

def envelope_detection(image):
    analytic_signal = hilbert(image)
    amplitude = np.abs(analytic_signal)
    return amplitude

def log_compression(image):
    dynamic_range=50
    compressed_image = 20 * np.log10(image / np.max(image))
    compressed_image = np.clip(compressed_image, -dynamic_range, 0) + dynamic_range
    return compressed_image

def scan_conversion(image):
    output_shape=(400, 400)
    zoom_factors = (output_shape[0] / image.shape[0], 
                    output_shape[1] / image.shape[1])
    converted_image = zoom(image, zoom_factors, order=1)
    return converted_image

for i in range(100):
    base, one_image = plt.subplots()

    image = das_beamforming(phantom_data, samples, sensors, position, sampling_rate, speed_of_sound_phantom, i)
    envelope_image = envelope_detection(image)
    compressed_image = log_compression(envelope_image)
    final_image = scan_conversion(compressed_image)
    
    one_image.imshow(final_image, cmap='gray')
    plt.savefig(f'frame_{i}.png')
    plt.close(base)

filenames = [f'frame_{i}.png' for i in range(100)]
with imageio.get_writer('video .gif', mode='I', fps=20) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
for filename in filenames:
    os.remove(filename)