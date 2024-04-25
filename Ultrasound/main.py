import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
from scipy.ndimage import zoom
from matplotlib.animation import FuncAnimation

rf_data = np.load('arm_rfdata.npy')
center_frequency = 7.6e6  # Hz
speed_of_sound = 1540  # m/s
speed_of_sound_phantom = 1450 # m/s
element_spacing = 0.3e-3  # m
sampling_rate = 31.16e6  # Hz
num_elements = 128  
num_samples = 2176  
element_pos = np.array([i * element_spacing for i in range(num_elements)])  


# 하나의 프레임에 대한 DAS beamforming
def das_beamforming(one_image, num_samples, num_elements, element_pos, sampling_rate, sound_speed, frame_idx):
    das_image = np.zeros((num_samples, num_elements))
    
    dt = 1 / sampling_rate # 샘플링 간격
    for i in range(num_samples):
        depth = i * sound_speed * dt / 2
        for j in range(num_elements):
            distance = np.sqrt(depth**2 + (element_pos[j] - element_pos[j//2])**2)
            time_delay = distance / sound_speed
            sample_index = int(time_delay / dt)
            # 샘플 인덱스가 존재하는 경우에만 값을 더함
            if sample_index < num_samples:
                das_image[i, j] += one_image[sample_index, j, frame_idx]
    return das_image

def envelope_detection(image):
    analytic_signal = hilbert(image)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def log_compression(image, dynamic_range=50):
    compressed_image = 20 * np.log10(image / np.max(image))
    compressed_image = np.clip(compressed_image, -dynamic_range, 0) + dynamic_range
    return compressed_image

def scan_conversion_single_frame(image, output_size=(400, 400)):
    if image.ndim != 2:
        raise ValueError("image must be a 2D array for a single frame.")
    zoom_factors = (output_size[0] / image.shape[0], output_size[1] / image.shape[1])
    converted_image = zoom(image, zoom_factors, order=1)
    return converted_image

# image = das_beamforming(rf_data, num_samples, num_elements, element_pos, sampling_rate, speed_of_sound, 99)
# env_img = envelope_detection(image)
# compressed_img = log_compression(env_img)
# final_img = scan_conversion_single_frame(compressed_img)




# # DAS beamforming
# image = das_beamforming(rf_data, element_pos, sampling_rate, speed_of_sound)

# fig, ax = plt.subplots()
# im = ax.imshow(np.zeros((400, 400)), cmap='gray')  # 초기 이미지

# def update(frame_number):
#     # 처리할 프레임 선택
#     frame_image = image[:, :, frame_number]
#     # 프레임에 대한 이미지 처리
#     envelope_image = envelope_detection(frame_image)
#     compressed_image = log_compression(envelope_image)
#     final_image = scan_conversion(compressed_image, (400, 400))
#     # 업데이트된 이미지로 설정
#     im.set_data(final_image)
#     return [im]

# # FuncAnimation을 사용하여 애니메이션 생성
# ani = FuncAnimation(fig, update, frames=image.shape[2], blit=True)
# plt.show()