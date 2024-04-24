import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
from scipy.ndimage import zoom
from matplotlib.animation import FuncAnimation

def das_beamforming(rf_data, element_pos, sampling_rate, sound_speed):
    num_samples, num_elements, num_frames = rf_data.shape
    image = np.zeros((num_samples, num_elements, num_frames))
    
    dt = 1 / sampling_rate
    
    for frame in range(num_frames):
        for i in range(num_samples):
            depth = i * sound_speed * dt / 2
            
            for j in range(num_elements):
                distance = np.sqrt(depth**2 + (element_pos[j] - element_pos[j//2])**2)
                time_delay = distance / sound_speed
                sample_index = int(time_delay / dt)
                
                if sample_index < num_samples:
                    image[i, j, frame] += rf_data[sample_index, j, frame]
                    
    return image

def envelope_detection(image):
    analytic_signal = hilbert(image)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def log_compression(image, dynamic_range=50):
    compressed_image = 20 * np.log10(image / np.max(image))
    compressed_image = np.clip(compressed_image, -dynamic_range, 0) + dynamic_range
    return compressed_image

def scan_conversion(image, output_size):
    desired_shape = (400, 400)
    original_shape = image.shape

    zoom_factors = (
        desired_shape[0] / original_shape[0],  # 세로 방향 확대 비율
        desired_shape[1] / original_shape[1],  # 가로 방향 확대 비율
        )
    num_frames = image.shape[2]
    converted_frames = []
    zoom_factors = (output_size[0] / image.shape[0], output_size[1] / image.shape[1])
    for frame in range(num_frames):
        # 각 프레임에 대해 zoom 적용
        converted_frame = zoom(image[:, :, frame], zoom_factors, order=1)
        converted_frames.append(converted_frame)
    # 모든 변환된 프레임을 하나의 3차원 배열로 합침
    converted_image = np.stack(converted_frames, axis=2)

    return converted_image

rf_data = np.load('arm_rfdata.npy')
center_frequency = 7.6e6  # Hz
speed_of_sound = 1540  # m/s
element_spacing = 0.3e-3  # m
sampling_rate = 31.16e6  # Hz
num_elements = 128  # Assuming the number of elements in your array
num_samples = 2176  # Number of samples per frame
element_pos = np.array([i * element_spacing for i in range(num_elements)]) # Generate positions for each element based on spacing


# DAS beamforming
image = das_beamforming(rf_data, element_pos, sampling_rate, speed_of_sound)

fig, ax = plt.subplots()
im = ax.imshow(np.zeros((400, 400)), cmap='gray')  # 초기 이미지

def update(frame_number):
    # 처리할 프레임 선택
    frame_image = image[:, :, frame_number]
    # 프레임에 대한 이미지 처리
    envelope_image = envelope_detection(frame_image)
    compressed_image = log_compression(envelope_image)
    final_image = scan_conversion(compressed_image, (400, 400))
    # 업데이트된 이미지로 설정
    im.set_data(final_image)
    return [im]

# FuncAnimation을 사용하여 애니메이션 생성
ani = FuncAnimation(fig, update, frames=image.shape[2], blit=True)
plt.show()