import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fftpack import fft, ifft, fftshift
from skimage.transform import radon, iradon

# .npy 파일 로드
sino = np.load('C:/Users/bt090/CT/sinogram.npy')
print('Data shape:', sino.shape)

sinogram = sino.T
print('Data shape:', sinogram.shape)

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
theta = np.linspace(0., 180., sinogram.shape[0], endpoint=False)

# Back Projection을 수행합니다.
reconstructed_img = back_projection(sinogram, theta)

# # 결과를 시각화합니다.
# plt.imshow(reconstructed_img, cmap='gray')
# plt.title('Back Projected Image')
# plt.axis('off')
# plt.show()

# Apply ramp filter to sinogram
filtered_sinogram = np.abs(np.fft.ifft(np.fft.fftshift(np.fft.fft(sinogram, axis=1), axes=1), axis=1))

# Perform back projection on filtered sinogram
filtered_reconstructed_img = back_projection(filtered_sinogram, theta)

# Display the filtered reconstructed image
plt.imshow(filtered_reconstructed_img, cmap='gray')
plt.title('Filtered Reconstructed Image')
plt.axis('off')
plt.show()