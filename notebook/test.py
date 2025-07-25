import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import math

image_path = r'D:\py\Digital_Image_Processing\Image\6.png'
I = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB) 

r, g, b = I[:,:,0] / 255.0, I[:,:,1] / 255.0, I[:,:,2] / 255.0

def gamma_correct(c):
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)

R = gamma_correct(r)
G = gamma_correct(g)
B = gamma_correct(b)

X = 0.4124 * R + 0.3567 * G + 0.1805 * B
Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
Z = 0.0193 * R + 0.1192 * G + 0.9505 * B


Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
x = X / Xn
y = Y / Yn
z = Z / Zn

def f(t):
    delta = 6/29
    return np.where(t > delta**3, t ** (1/3), (t / (3 * delta**2)) + 4/29)

fx = f(x)
fy = f(y)
fz = f(z)

L = 116 * fy - 16
a = 500 * (fx - fy)
b = 200 * (fy - fz)

L = np.zeros_like(fy)

lab_image = np.stack([L, a, b], axis=-1)

L_vis = np.clip(L * (255 / 100), 0, 255)
a_vis = np.clip(a + 128, 0, 255)
b_vis = np.clip(b + 128, 0, 255)
lab_vis = np.stack([L_vis, a_vis, b_vis], axis=-1).astype(np.uint8)

def kmeans_manual (data, K,  max_iters = 100, tol = 1e-4):

    N, D = data.shape

    np.random.seed(42) # Make the random is same with other time 
    initial_indices = np.random.choice(N, K, replace = False) # No same value for same point
    centeroids = data[initial_indices]
    
    for iter in range(max_iters):
        # Calculate distance and labels
        distance = np.linalg.norm(data[:, np.newaxis] - centeroids, axis=2)
        labels = np.argmin(distance, axis=1)

        new_centeroid = np.zeros((K, D))
        for k in range(K):
            clustered_point = data[labels == k]
            if len(clustered_point) > 0:
                new_centeroid[k] = np.mean(clustered_point, axis = 0)
            else:
                new_centeroid[k] = data[np.random.choice(N)]

        if np.linalg.norm(new_centeroid - centeroids) < tol:
            break
        
        centeroids = new_centeroid

    return labels, centeroids

ab = np.stack([a, b], axis=-1)
h, w, _ = ab.shape
ab_flat = ab.reshape((-1, 2))

labels, centroids = kmeans_manual(ab_flat, K=3)

segmentation = labels.reshape((h, w))

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(I)
plt.title("Original RGB Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(lab_vis, cmap="gray")
plt.title("Converted L*a*b* Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(segmentation, cmap='gray')
plt.axis('off')
plt.title(f"Final Segmentation with K={K}")