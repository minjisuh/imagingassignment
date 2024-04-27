import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy.fft as fft
from scipy.fftpack import fft, ifft, fftfreq, fftshift

# .npy 파일 로드
sinogram = np.load('C:/Users/bt090/CT/sinogram.npy')
print('Data shape:', sinogram.shape)

def back_projection(sinogram, theta):
    # sinogram의 행은 레이더 센서의 인덱스, 열은 각도입니다.
    num_sensors, num_angles = sinogram.shape
    # 출력 이미지의 크기를 정합니다. 정사각형 이미지로 가정합니다.
    output_size = sinogram.shape[0]
    output_img = np.zeros((output_size, output_size), dtype=np.float32)
    
    # 각도에 따른 back projection을 수행합니다.
    for i in range(num_angles):
        # 현재 각도에서의 sinogram 데이터
        projection = sinogram[:, i]
        
        # 해당 각도에서의 이미지를 생성합니다.
        # np.newaxis를 사용하여 배열의 차원을 조정합니다.
        projection_img = np.tile(projection[:, np.newaxis], (1, output_size))
        
        # 이미지를 원래 각도로 회전시킵니다.
        projection_img = ndimage.rotate(projection_img, theta[i], reshape=False, prefilter=True)
        
        # 회전된 이미지를 결과 이미지에 더합니다.
        output_img += projection_img
        
    # 모든 각도에 대한 평균을 취합니다.
    output_img /= num_angles
    return output_img

# Sinogram 데이터와 각도 벡터를 로드합니다.
theta = np.linspace(0., 180., sinogram.shape[1], endpoint=False)

# Back Projection을 수행합니다.
reconstructed_img = back_projection(sinogram, theta)
plt.imshow(reconstructed_img, cmap = 'gray')
plt.show()

def create_filter(num_pixels, filter_type='ram-lak', cutoff=10.0):
    # 필터 생성
    freqs = fftfreq(num_pixels).reshape(-1, 1)
    filter = 2 * np.abs(freqs)  # Ram-Lak 필터 기본 형태
    if filter_type == 'shepp-logan':
        filter *= np.sinc(freqs / (2 * cutoff))
    elif filter_type == 'cosine':
        filter *= np.cos(np.pi * freqs / (2 * cutoff))
    filter = fftshift(filter)  # 필터를 주파수 영역으로 이동
    return filter.flatten()

def apply_filter(sinogram):
    # 시노그램에 필터 적용
    num_pixels, num_projections = sinogram.shape
    filter = create_filter(num_pixels)
    filtered_sinogram = np.zeros_like(sinogram)
    for i in range(num_projections):
        projection = sinogram[:, i]
        fft_projection = fftshift(fft(projection))
        filtered_projection = fft_projection * filter
        filtered_sinogram[:, i] = np.real(ifft(fftshift(filtered_projection)))
    return filtered_sinogram

filtered_sinogram = apply_filter(sinogram)
reconstructed_image = back_projection2(filtered_sinogram, theta)
plt.imshow(reconstructed_image, cmap = 'gray')
plt.show()