import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

FL = np.array ([[-2, 3, -6, 3, -2],
                [ 3, 4, 2, 4, 3],
                [-6, 2, 48, 2, -6],
                [3, 4, 2, 4, 3],
                [-2, 3, -6, 3, -2]]) / 64

fft_FL = np.fft.fft2(FL)
shift_fft_FL = np.fft.fftshift(fft_FL)
magnitude_FL = np.abs(shift_fft_FL)
log_magnitude = np.log(1 + magnitude_FL)

plt.figure(figsize=(12, 10))
plt.imshow(magnitude_FL, cmap='gray' , extent=[-0.5, 0.5, -0.5, 0.5])
plt.title("Magnitude Spectrum")

plt.colorbar()
plt.show()