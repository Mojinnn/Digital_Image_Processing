import cv2
import numpy as np
import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from scipy.signal import convolve2d
import re

# Paths
image_folder = "D:/py/Digital_Image_Processing/Image/"
output_file = "D:/py/Digital_Image_Processing/evaluation.xlsx"

# Function to compute PSNR
def compute_psnr(original, reconstructed):
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100  # No difference
    
    psnr = 10 * np.log10((255**2) / mse)
    return psnr

# Function to compute SSIM
def compute_ssim(original, reconstructed):
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    reconstructed_gray = cv2.cvtColor(reconstructed, cv2.COLOR_RGB2GRAY)

    return ssim(original_gray, reconstructed_gray, data_range=255)

# Function to create a mosaicked image (Bayer filter)
def create_mosaic(I):
    r, g, b = cv2.split(I)
    
    r[1::2, :] = 0
    r[:, 1::2] = 0
    
    g[::2, ::2] = 0
    g[1::2, 1::2] = 0
    
    b[::2, :] = 0
    b[:, ::2] = 0

    return cv2.merge([r, g, b])

# Function for Bilinear Interpolation
def bilinear_reconstruction(I):
    mosaic = create_mosaic(I)
    r, g, b = cv2.split(mosaic)

    F_rb = np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]]) / 4

    F_g = np.array([[0, 1, 0],
                    [1, 4, 1],
                    [0, 1, 0]]) / 4

    r_re = convolve2d(r, F_rb, 'same', 'symm')
    g_re = convolve2d(g, F_g, 'same', 'symm')
    b_re = convolve2d(b, F_rb, 'same', 'symm')

    return cv2.merge([r_re, g_re, b_re]).astype(np.uint8)

# Function for Alleys Method
def alleys_reconstruction(I):
    h, w, t = I.shape

    multiplexed_image = np.zeros((h, w))
    multiplexed_image[::2, ::2] = I[::2, ::2, 0]  # Red
    multiplexed_image[::2, 1::2] = I[::2, 1::2, 1]  # Green
    multiplexed_image[1::2, ::2] = I[1::2, ::2, 1]  # Green
    multiplexed_image[1::2, 1::2] = I[1::2, 1::2, 2]  # Blue

    FL = np.array([[-2, 3, -6, 3, -2],
                   [3, 4, 2, 4, 3],
                   [-6, 2, 48, 2, -6],
                   [3, 4, 2, 4, 3],
                   [-2, 3, -6, 3, -2]]) / 64
    estimate_luminance = convolve2d(multiplexed_image, FL, mode='same', boundary='symm')

    multiplexed_chrominance = multiplexed_image - estimate_luminance

    mR, mG, mB = np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))
    mR[::2, ::2] = 1
    mG[::2, 1::2] = 1
    mG[1::2, ::2] = 1
    mB[1::2, 1::2] = 1

    sub_chro_red = multiplexed_chrominance * mR
    sub_chro_green = multiplexed_chrominance * mG
    sub_chro_blue = multiplexed_chrominance * mB

    F_rb_2 = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 4
    F_g_2 = np.array([[0, 1, 0],
                      [1, 4, 1],
                      [0, 1, 0]]) / 4

    interpolated_red = convolve2d(sub_chro_red, F_rb_2, mode='same', boundary='symm')
    interpolated_green = convolve2d(sub_chro_green, F_g_2, mode='same', boundary='symm')
    interpolated_blue = convolve2d(sub_chro_blue, F_rb_2, mode='same', boundary='symm')

    reconstructed_image = np.zeros_like(I, dtype=np.float64)
    reconstructed_image[:, :, 0] = interpolated_red + estimate_luminance
    reconstructed_image[:, :, 1] = interpolated_green + estimate_luminance
    reconstructed_image[:, :, 2] = interpolated_blue + estimate_luminance

    return np.clip(reconstructed_image, 0, 255).astype(np.uint8)


# Function for Laroche method
def laroche_reconstruction(I):
    I = I.astype(np.float64)
    h, w, c = I.shape

    # Tạo ảnh có padding 2 pixel mỗi bên
    I_padded = np.zeros((h + 4, w + 4, c))
    I_padded[2:h+2, 2:w+2, :] = I

    I_R = I_padded[:, :, 0]
    I_G = I_padded[:, :, 1]
    I_B = I_padded[:, :, 2]

    H_img, W, _ = I_padded.shape

    # Khởi tạo ảnh Bayer CFA
    lr = np.zeros_like(I_padded)

    lr[2:H_img-2:2, 2:W-2:2, 0] = I_R[2:H_img-2:2, 2:W-2:2]  # Red
    lr[3:H_img-2:2, 3:W-2:2, 2] = I_B[3:H_img-2:2, 3:W-2:2]  # Blue
    lr[3:H_img-2:2, 2:W-2:2, 1] = I_G[3:H_img-2:2, 2:W-2:2]  # Green
    lr[2:H_img-2:2, 3:W-2:2, 1] = I_G[2:H_img-2:2, 3:W-2:2]  # Green

    # Nội suy màu G tại pixel R
    for i in range(2, H_img-2, 2):
        for j in range(2, W-2, 2):
            deltaH1 = (lr[i-2, j, 0] + lr[i+2, j, 0]) / 2
            deltaV1 = (lr[i, j-2, 0] + lr[i, j+2, 0]) / 2
            if deltaH1 < deltaV1:
                lr[i, j, 1] = (lr[i-1, j, 1] + lr[i+1, j, 1]) / 2
            elif deltaH1 > deltaV1:
                lr[i, j, 1] = (lr[i, j-1, 1] + lr[i, j+1, 1]) / 2
            else:
                lr[i, j, 1] = (lr[i-1, j, 1] + lr[i+1, j, 1] + lr[i, j-1, 1] + lr[i, j+1, 1]) / 4

    # Nội suy màu G tại pixel B
    for i in range(3, H_img-2, 2):
        for j in range(3, W-2, 2):
            deltaH1 = (lr[i-2, j, 2] + lr[i+2, j, 2]) / 2
            deltaV1 = (lr[i, j-2, 2] + lr[i, j+2, 2]) / 2
            if deltaH1 < deltaV1:
                lr[i, j, 1] = (lr[i-1, j, 1] + lr[i+1, j, 1]) / 2
            elif deltaH1 > deltaV1:
                lr[i, j, 1] = (lr[i, j-1, 1] + lr[i, j+1, 1]) / 2
            else:
                lr[i, j, 1] = (lr[i-1, j, 1] + lr[i+1, j, 1] + lr[i, j-1, 1] + lr[i, j+1, 1]) / 4

    # Nội suy màu B tại pixel R
    for i in range(2, H_img-2, 2):
        for j in range(2, W-2, 2):
            lr[i, j, 2] = (lr[i-1, j-1, 2] - lr[i-1, j-1, 1] +
                           lr[i+1, j-1, 2] - lr[i+1, j-1, 1] +
                           lr[i+1, j+1, 2] - lr[i+1, j+1, 1] +
                           lr[i-1, j+1, 2] - lr[i-1, j+1, 1]) / 4 + lr[i, j, 1]

    # Nội suy màu R tại pixel B
    for i in range(3, H_img-2, 2):
        for j in range(3, W-2, 2):
            lr[i, j, 0] = (lr[i-1, j-1, 0] - lr[i-1, j-1, 1] +
                           lr[i+1, j-1, 0] - lr[i+1, j-1, 1] +
                           lr[i+1, j+1, 0] - lr[i+1, j+1, 1] +
                           lr[i-1, j+1, 0] - lr[i-1, j+1, 1]) / 4 + lr[i, j, 1]

    # Nội suy màu B và R tại pixel G
    for i in range(3, H_img-2, 2):
        for j in range(2, W-2, 2):
            lr[i, j, 0] = (lr[i, j-1, 0] - lr[i, j-1, 1] + lr[i, j+1, 0] - lr[i, j+1, 1]) / 2 + lr[i, j, 1]
            lr[i, j, 2] = (lr[i, j-1, 2] - lr[i, j-1, 1] + lr[i, j+1, 2] - lr[i, j+1, 1]) / 2 + lr[i, j, 1]

    for i in range(2, H_img-2, 2):
        for j in range(3, W-2, 2):
            lr[i, j, 0] = (lr[i, j-1, 0] - lr[i, j-1, 1] + lr[i, j+1, 0] - lr[i, j+1, 1]) / 2 + lr[i, j, 1]
            lr[i, j, 2] = (lr[i, j-1, 2] - lr[i, j-1, 1] + lr[i, j+1, 2] - lr[i, j+1, 1]) / 2 + lr[i, j, 1]

    # Cắt lại ảnh để loại bỏ padding
    lr_cropped = lr[2:h+2, 2:w+2, :]
    
    return np.clip(lr_cropped, 0, 255).astype(np.uint8)

# Function for Edge-directed method
def edge_directed_reconstruction(I):

    r, g, b = cv2.split(I)
    r[1::2, :] = 0
    r[:, 1::2] = 0
    g[::2, 1::2] = 0
    g[1::2, ::2] = 0
    b[::2, :] = 0
    b[:, ::2] = 0

    e = 0

    h, w = g.shape

    g_padded = np.zeros((h + 2, w + 2), dtype=np.float64)
    g_padded[1:-1, 1:-1] = g

    for i in range(1, h, 2):
        for j in range(2, w, 2):
            G1, G2, G3, G4 = g_padded[i-1, j], g_padded[i, j-1], g_padded[i+1, j], g_padded[i, j+1]
            delta_H = abs(G2 - G3)
            delta_V = abs(G1 - G4)
            if delta_H + e < delta_V:
                g_padded[i, j] = (G2 + G3) / 2
            elif delta_V + e < delta_H:
                g_padded[i, j] = (G1 + G4) / 2
            else:
                g_padded[i, j] = (G1 + G2 + G3 + G4) / 4

    for i in range(2, h, 2):
        for j in range(1, w, 2):
            G1, G2, G3, G4 = g_padded[i-1, j], g_padded[i, j-1], g_padded[i+1, j], g_padded[i, j+1]
            delta_H = abs(G2 - G3)
            delta_V = abs(G1 - G4)
            if delta_H + e < delta_V:
                g_padded[i, j] = (G2 + G3) / 2
            elif delta_V + e < delta_H:
                g_padded[i, j] = (G1 + G4) / 2
            else:
                g_padded[i, j] = (G1 + G2 + G3 + G4) / 4

    g = g_padded[1:-1, 1:-1]
    reconstructed_green = g.astype(np.float64)

    F_rb = np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]]) / 4


    mR, mG, mB = np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))
    mR[::2, ::2] = 1
    mG[::2, 1::2] = 1
    mG[1::2, ::2] = 1
    mB[1::2, 1::2] = 1
    
    r_d = r - reconstructed_green
    r_d1 = r_d * mR
    r_d2 = convolve2d(r_d1, F_rb, 'same', 'symm')
    reconstructed_red = (r_d2 + reconstructed_green).astype(np.float64)

    b_d = b - reconstructed_green
    b_d1 = b_d * mB
    b_d2 = convolve2d(b_d1, F_rb, 'same', 'symm')
    reconstructed_blue = (b_d2 + reconstructed_green).astype(np.float64)

    reconstructed_image = np.zeros_like(I, dtype=np.float64)

    reconstructed_image[:, :, 0] = np.clip(reconstructed_red, 0, 255)
    reconstructed_image[:, :, 1] = np.clip(reconstructed_green, 0, 255)
    reconstructed_image[:, :, 2] = np.clip(reconstructed_blue, 0, 255)

    return np.clip(reconstructed_image, 0, 255).astype(np.uint8)

# Sort files numerically
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

image_list = sorted(os.listdir(image_folder), key=extract_number)[:24]

results = []

for image_name in image_list:
    image_path = os.path.join(image_folder, image_name)
    
    I = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if I is None:
        print(f"⚠️ Warning: Could not load {image_name}. Skipping...")
        continue
    
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    bilinear_image = bilinear_reconstruction(I)
    alleys_image = alleys_reconstruction(I)
    edge_directed_image = edge_directed_reconstruction(I)
    laroche_image = laroche_reconstruction(I)
    
    psnr_bilinear = compute_psnr(I, bilinear_image)
    ssim_bilinear = compute_ssim(I, bilinear_image)
    psnr_alleys = compute_psnr(I, alleys_image)
    ssim_alleys = compute_ssim(I, alleys_image)
    psnr_edge_directed = compute_psnr(I, edge_directed_image)
    ssim_edge_directed = compute_ssim(I, edge_directed_image)
    psnr_laroche = compute_psnr(I, laroche_image)
    ssim_laroche = compute_ssim(I, laroche_image)

    results.append([image_name, psnr_bilinear, ssim_bilinear, psnr_alleys, ssim_alleys, psnr_edge_directed, ssim_edge_directed, psnr_laroche, ssim_laroche])

# Compute averages
avg_psnr_bilinear = np.mean([row[1] for row in results])
avg_ssim_bilinear = np.mean([row[2] for row in results])
avg_psnr_alleys = np.mean([row[3] for row in results])
avg_ssim_alleys = np.mean([row[4] for row in results])
avg_psnr_edge_directed = np.mean([row[5] for row in results])
avg_ssim_edge_directed = np.mean([row[6] for row in results])
avg_psnr_laroche = np.mean([row[7] for row in results])
avg_ssim_laroche = np.mean([row[8] for row in results])

# Append averages to results
results.append(["Average", avg_psnr_bilinear, avg_ssim_bilinear, avg_psnr_alleys, avg_ssim_alleys, avg_psnr_edge_directed, avg_ssim_edge_directed, avg_psnr_laroche, avg_ssim_laroche])

# Save to Excel
df = pd.DataFrame(results, columns=["Image_Name", "PSNR_Bilinear", "SSIM_Bilinear", "PSNR_Alleys", "SSIM_Alleys", "PSNR_Edge_Directed", "SSIM_Edge_Directed", "PSNR_Laroche", "SSIM_Laroche"])
df.to_excel(output_file, index=False)

print(f"✅ Evaluation complete! Results saved to '{output_file}'.")
