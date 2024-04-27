import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy.fft as fft
from scipy.fftpack import fft, ifft, fftfreq, fftshift

#  주어진 sinogram 파일 로드
sinogram = np.load('CT/sinogram.npy')
pixels, angles = sinogram.shape
theta = list(range(0, 180)) # 0~179도의 각도 리스트 생성
# plt.imshow(sinogram, cmap = 'gray')
# plt.title('Sinogram')
# plt.show()



# BP 함수 정의
def back_projection(pixels, angles, sinogram, theta):
    # output image의 shape에 맞는 512x512의 0의 행렬 생성
    base = np.zeros((512, 512), dtype=np.float32)
    
    for i in range(angles):
        projection = sinogram[:, i]  #현재 각도의 projection
        extended_projection = np.tile(projection, (512, 1))  # projection을 512x1로 확장
        extended_projection = ndimage.rotate(extended_projection, theta[i], reshape=False, prefilter=True)         # 이미지를 현재 각도만큼 회전
        base += extended_projection  # 회전된 이미지를 base에 더함
    return base

backprojected_img = back_projection(pixels, angles, sinogram, theta)
# plt.imshow(backprojected_img, cmap = 'gray')
# plt.title('Backprojected Image')
# plt.show()




# FBP 과정

# Ram-Lak 필터함수 정의 
def Ram_Sak_filter(pixels):
    sample_freq = fftfreq(pixels)
    filter = 2 * np.abs(sample_freq)  # Ram-Lak 필터 생성
    filter = fftshift(filter)  
    return filter.flatten()

def apply_filter(pixels, angles, sinogram):
    filter = Ram_Sak_filter(pixels)
    base_filtered = np.zeros((512, 180), dtype=np.float32)  # sinogram과 같은 모양으로 0으로 채운 행렬 생성
    for i in range(angles):
        projection = sinogram[:, i]  # 현재 각도의 projection
        fft_projection = fftshift(fft(projection))  # projection을 주파수 영역으로 변환
        filtered_projection = fft_projection * filter  # 필터 적용
        base_filtered[:, i] = np.real(ifft(fftshift(filtered_projection)))  # 필터링된 projection을 다시 이미지 영역으로 변환
    return base_filtered

base_filtered = apply_filter(pixels, angles, sinogram)
# plt.imshow(base_filtered, cmap = 'gray')
# plt.title('Filtered Sinogram')
# plt.show()


reconstructed_image = back_projection(pixels, angles, base_filtered, theta)
# plt.imshow(reconstructed_image, cmap = 'gray')
# plt.title('FBP Image filter = Ram-Lak')
# plt.show()




# Shepp-logan 필터함수 정의
def Shepp_logan_filter(pixels):
    sample_freq = fftfreq(pixels)
    filter = 2 * np.abs(sample_freq) * np.sinc(sample_freq) # Ram=Lak 필터에 sinc 함수를 곱하여 Shepp-logan 필터 생성
    filter = fftshift(filter) 
    return filter.flatten()

# Cosine 필터함수 정의
def Cosine_filter(pixels):
    sample_freq = fftfreq(pixels)
    filter = 2 * np.abs(sample_freq) * np.cos(np.pi * sample_freq)  # Ram-Lak 필터에 cos 함수를 곱하여 Cosine 필터 생성
    filter = fftshift(filter) 
    return filter.flatten()





# Shepp-logan 필터 적용
def apply_filter(pixels, angles, sinogram):
    filter = Shepp_logan_filter(pixels)
    base_filtered = np.zeros((512, 180), dtype=np.float32)
    for i in range(angles):
        projection = sinogram[:, i]
        fft_projection = fftshift(fft(projection))
        filtered_projection = fft_projection * filter
        base_filtered[:, i] = np.real(ifft(fftshift(filtered_projection)))
    return base_filtered

base_filtered_sl = apply_filter(pixels, angles, sinogram)
reconstructed_image_sl = back_projection(pixels, angles, base_filtered_sl, theta)
# plt.imshow(reconstructed_image_sl, cmap = 'gray')
# plt.title('FBP Image filter = Shepp-logan')
# plt.show()


# Cosine 필터 적용
def apply_filter(pixels, angles, sinogram):
    filter = Cosine_filter(pixels)
    base_filtered = np.zeros((512, 180), dtype=np.float32)
    for i in range(angles):
        projection = sinogram[:, i]
        fft_projection = fftshift(fft(projection))
        filtered_projection = fft_projection * filter
        base_filtered[:, i] = np.real(ifft(fftshift(filtered_projection)))
    return base_filtered

base_filtered_c = apply_filter(pixels, angles, sinogram)
reconstructed_image_c = back_projection(pixels, angles, base_filtered_c, theta)
plt.imshow(reconstructed_image_c, cmap = 'gray')
plt.title('FBP Image filter = Cosine')
plt.show()