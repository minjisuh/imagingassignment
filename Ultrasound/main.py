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
position = np.array([i * spacing for i in range(sensors)])  # spacing 간격만큼 증가하는 배열 생성


# DAS 과정 수행
def das(rf_data, samples, sensors, position, sampling_rate, speed_of_sound, frame_number):
    base_das = np.zeros((samples, sensors)) # 2176 x 128
    for i in range(samples):
        depth = i * speed_of_sound / (2 * sampling_rate) # 깊이 계산 
        for j in range(sensors):
            distance = np.sqrt(depth**2 + (position[j] - position[sensors//2])**2) # 중앙센서와의 거리 계산
            time_delay = distance / speed_of_sound # 시간 지연 계산
            sample_index = int(time_delay *sampling_rate) # 샘플 인덱스 계산
            base_das[i, j] += rf_data[sample_index, j, frame_number] # base_das에 rf_data 값 저장
    return base_das

# Envelope Detection 과정 수행
def envelope_detection(image):
    analytic_signal = hilbert(image) # 헤일버트 변환
    amplitude = np.abs(analytic_signal) # 절대값 계산
    return amplitude

# Log Compression 과정 수행
def log_compression(image):
    dynamic_range=50
    compressed_image = 20 * np.log10(image / np.max(image)) # 로그 압축, dB로 변환
    compressed_image = np.clip(compressed_image, -dynamic_range, 0) + dynamic_range # 동적 범위 조정
    return compressed_image

# Scan Conversion 과정 수행
def scan_conversion(image):
    output_shape=(400, 400)
    zoom_factors = (output_shape[0] / image.shape[0], 
                    output_shape[1] / image.shape[1]) 
    converted_image = zoom(image, zoom_factors, order=1) # 이미지 크기 조정
    return converted_image

# arm_rfdata로 영상 생성
# frame별 이미지 생성
for i in range(100):
    base, one_image = plt.subplots()
    image = das(rf_data, samples, sensors, position, sampling_rate, speed_of_sound, i)
    envelope_image = envelope_detection(image)
    compressed_image = log_compression(envelope_image)
    final_image = scan_conversion(compressed_image)
    
    one_image.imshow(final_image, cmap='gray')
    plt.savefig(f'frame_{i}.png') # 한 frame의 이미지 저장
    plt.close(base)

filenames = [f'frame_{i}.png' for i in range(100)]
with imageio.get_writer('arm_rfdata 초음파 영상.gif', mode='I', fps=20) as writer: # gif 파일 생성
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
for filename in filenames: # 이미지 파일 삭제
    os.remove(filename)



# phantom_data로 영상 생성
# frame별 이미지 생성
for i in range(100):
    base, one_image = plt.subplots()
    image = das(phantom_data, samples, sensors, position, sampling_rate, speed_of_sound_phantom, i)
    envelope_image = envelope_detection(image)
    compressed_image = log_compression(envelope_image)
    final_image = scan_conversion(compressed_image)
    
    one_image.imshow(final_image, cmap='gray')
    plt.savefig(f'frame_{i}.png') # 한 frame의 이미지 저장
    plt.close(base)

filenames = [f'frame_{i}.png' for i in range(100)]
with imageio.get_writer('phantom 초음파 영상.gif', mode='I', fps=20) as writer: # gif 파일 생성
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
for filename in filenames: # 이미지 파일 삭제
    os.remove(filename)