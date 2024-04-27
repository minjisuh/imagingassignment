import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy.fft as fft
from scipy.fftpack import fft, ifft, fftfreq, fftshift


# .npy 파일 로드
sinogram = np.load('C:/Users/bt090/CT/sinogram.npy')
print('Data shape:', sinogram.shape)

sinograms = sinogram.T
print('Data shape:', sinograms.shape)

# sinogram 이미지 출력
# plt.imshow(sinograms, cmap = 'gray')
# plt.show()


def back_projection(sinograms, theta):
    # sinogram의 행은 각도, 열은 레이더 센서의 인덱스입니다.
    num_angles, num_sensors = sinograms.shape
    # 출력 이미지의 크기를 정합니다. 정사각형 이미지로 가정합니다.
    output_size = sinograms.shape[1]
    output_img = np.zeros((output_size, output_size), dtype=np.float32)
    
    # 각도에 따른 back projection을 수행합니다.
    for i in range(num_angles):
        # 현재 각도에서의 sinogram 데이터
        projection = sinograms[i, :]
        
        # 해당 각도에서의 이미지를 생성합니다.
        # np.newaxis를 사용하여 배열의 차원을 조정합니다.
        projection_img = np.tile(projection, (output_size, 1))
        
        # 이미지를 원래 각도로 회전시킵니다.
        projection_img = ndimage.rotate(projection_img, theta[i], reshape=False, prefilter=True)
        
        # 회전된 이미지를 결과 이미지에 더합니다.
        output_img += projection_img
        
    # 모든 각도에 대한 평균을 취합니다.
    output_img /= num_angles
    return output_img

# Sinogram 데이터와 각도 벡터를 로드합니다.
# 이 부분은 실제 데이터에 맞게 조정해야 합니다.
theta = np.linspace(0., 180., sinograms.shape[0], endpoint=False)

# Back Projection을 수행합니다.
reconstructed_img = back_projection(sinograms, theta)

# 결과를 시각화합니다.
# plt.imshow(reconstructed_img, cmap='gray')
# plt.title('Back Projected Image')
# plt.axis('off')
# plt.show()

def create_filter(num_pixels, filter_type='ram-lak', cutoff=10.0):
    # 필터 생성
    freqs = fftfreq(num_pixels).reshape(-1, 1)
    filter = 2 * np.abs(freqs)  # Ram-Lak 필터 기본 형태
    if filter_type == 'shepp-logan':
        # Shepp-Logan 필터는 Ram-Lak 필터에 사인 함수를 곱함
        filter *= np.sinc(freqs / (2 * cutoff))
    elif filter_type == 'cosine':
        # 코사인 필터
        filter *= np.cos(np.pi * freqs / (2 * cutoff))
    filter = fftshift(filter)  # 필터를 주파수 영역으로 이동
    return filter.flatten()

def apply_filter(sinogram):
    # 시노그램에 필터 적용
    num_projections, num_pixels = sinogram.shape
    filter = create_filter(num_pixels)
    filtered_sinogram = np.zeros_like(sinogram)
    for i in range(num_projections):
        projection = sinogram[i, :]
        fft_projection = fftshift(fft(projection))
        filtered_projection = fft_projection * filter
        filtered_sinogram[i, :] = np.real(ifft(fftshift(filtered_projection)))
    return filtered_sinogram


filtered_sinogram = apply_filter(sinograms)
reconstructed_image = back_projection(filtered_sinogram, theta)
plt.imshow(reconstructed_image, cmap = 'gray')
plt.show()