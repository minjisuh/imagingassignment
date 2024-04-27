import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
from scipy.ndimage import zoom
import imageio.v2 as imageio
import os

rf_data = np.load('Ultrasound/arm_rfdata.npy')
center_frequency = 7.6e6  # Hz
sound_speed = 1450  # m/s
speed_of_sound_phantom = 1450 # m/s
element_spacing = 0.3e-3  # m
sampling_rate = 31.16e6  # Hz
num_elements = 128  
num_samples = 2176  
element_pos = np.array([i * element_spacing for i in range(num_elements)])  


# 하나의 프레임에 대한 DAS beamforming
def das_beamforming(rf_data, num_samples, num_elements, element_pos, sampling_rate, sound_speed, frame_number):
    das_image = np.zeros((num_samples, num_elements))
    
    for i in range(num_samples):
        depth = i * sound_speed / (2 * sampling_rate) 
        for j in range(num_elements):
            distance = np.sqrt(depth**2 + (element_pos[j] - element_pos[num_elements//2])**2)
            time_delay = distance / sound_speed
            sample_index = int(time_delay *sampling_rate)
            if sample_index < num_samples:
                das_image[i, j] += rf_data[sample_index, j, frame_number]
    return das_image

def envelope_detection(image):
    analytic_signal = hilbert(image)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def log_compression(image, dynamic_range=50):
    compressed_image = 20 * np.log10(image / np.max(image))
    compressed_image = np.clip(compressed_image, -dynamic_range, 0) + dynamic_range
    return compressed_image

def scan_conversion(image, output_size=(400, 400)):
    zoom_factors = (output_size[0] / image.shape[0], output_size[1] / image.shape[1])
    converted_image = zoom(image, zoom_factors, order=1)
    return converted_image

for i in range(100):
    fig, ax = plt.subplots()
    image = das_beamforming(rf_data, num_samples, num_elements, element_pos, sampling_rate, sound_speed, i)
    envelope_image = envelope_detection(image)
    compressed_image = log_compression(envelope_image)
    final_image = scan_conversion(compressed_image, (400, 400))
    ax.imshow(final_image, cmap='gray')
    plt.savefig(f'frame_{i}.png')
    plt.close(fig)

filenames = [f'frame_{i}.png' for i in range(100)]
with imageio.get_writer('video .gif', mode='I', fps=20) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
for filename in filenames:
    os.remove(filename)