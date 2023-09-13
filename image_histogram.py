from typing import List, Union

import cv2
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

def calculate_histogram(image: NDArray) -> List[int]:
    histogram = [0] * 256
    
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            histogram[image[row, col]] += 1
            
    return histogram

def add_gaussian_noise(image: NDArray, mean: int = 0, std_dev: int = 50) -> NDArray:
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def mse(original_image: NDArray, distorted_image: NDArray) -> float:
    err = np.sum((original_image.astype("float") - distorted_image.astype("float")) ** 2)
    err /= float(original_image.shape[0] * original_image.shape[1])
    return err

def psnr(original_image, distorted_image, max_pixel_value=255.0) -> Union[float, str]:
    mean_squared_err = mse(original_image, distorted_image)
    if mean_squared_err == 0:
        return "Same Image"
    return 20 * np.log10(max_pixel_value / np.sqrt(mean_squared_err))

def modified_psnr(original_image, distorted_image, max_pixel_value=255.0) -> Union[float, str]:
    mean_squared_err = mse(original_image, distorted_image)
    if mean_squared_err == 0:
        return "Same Image"
    return 10 * np.log10(max_pixel_value / mean_squared_err)

def apply_mask(image: NDArray, mask: NDArray, multiplier: float = 1.0) -> NDArray:
    output_image = np.zeros_like(image)
    image_height, image_width = image.shape
    mask_height, mask_width = mask.shape
    mask_center_x = mask_width // 2
    mask_center_y = mask_height // 2

    for y in range(mask_center_y, image_height - mask_center_y):
        for x in range(mask_center_x, image_width - mask_center_x):
            roi = image[y - mask_center_y:y + mask_center_y + 1, x - mask_center_x:x + mask_center_x + 1]
            convolution_result = np.sum(roi * mask) * multiplier
            output_image[y, x] = convolution_result

    return output_image


images_list = ['lena.tif', 'camera.tif', 'moon.tiff']
mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
mask_multiplier = 1/9

for image_file in images_list:

    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    original_histogram = calculate_histogram(image)

    noisy_image_20 = add_gaussian_noise(image, mean=0, std_dev=20)
    noisy_histogram_20 = calculate_histogram(noisy_image_20)
    psnr_noisy_20 = psnr(image, noisy_image_20)
    print(f"PSNR Imagem Ruído Nível 1: {psnr_noisy_20}")

    noisy_image_50 = add_gaussian_noise(image, mean=0, std_dev=50)
    noisy_histogram_50 = calculate_histogram(noisy_image_50)
    masked_noisy_50 = apply_mask(noisy_image_50, mask, mask_multiplier)
    psnr_noisy_50 = psnr(image, noisy_image_50)
    print(f"PSNR Imagem Ruído Nível 2: {psnr_noisy_50}")

    noisy_image_80 = add_gaussian_noise(image, mean=0, std_dev=80)
    noisy_histogram_80 = calculate_histogram(noisy_image_80)
    psnr_noisy_80 = psnr(image, noisy_image_80)
    print(f"PSNR Imagem Ruído Nível 3: {psnr_noisy_80}")

    plt.figure(figsize=(10, 12))

    # Original Image
    plt.subplot(4, 4, 1)
    plt.title('Original Image')
    plt.axis('off')
    plt.imshow(image, cmap='gray')

    # Original Image Histogram
    plt.subplot(4, 4, 2)
    plt.title('Histogram (Original)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.bar(range(256), original_histogram)

    # Noisy Image 20
    plt.subplot(4, 4, 3)
    plt.title('Noisy Image 20')
    plt.axis('off')
    plt.imshow(noisy_image_20, cmap='gray')

    # Noisy Image 20 Histogram
    plt.subplot(4, 4, 4)
    plt.title('Histogram (Noisy 20)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.bar(range(256), noisy_histogram_20)

    # Noisy Image 50
    plt.subplot(4, 4, 5)
    plt.title('Noisy Image 50')
    plt.axis('off')
    plt.imshow(noisy_image_50, cmap='gray')

    # Noisy Image 50 Histogram
    plt.subplot(4, 4, 6)
    plt.title('Histogram (Noisy 50)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.bar(range(256), noisy_histogram_50)

    # Noisy Image 80
    plt.subplot(4, 4, 7)
    plt.title('Noisy Image 80')
    plt.axis('off')
    plt.imshow(noisy_image_80, cmap='gray')

    # Noisy Image 80 Histogram
    plt.subplot(4, 4, 8)
    plt.title('Histogram (Noisy 80)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.bar(range(256), noisy_histogram_80)

    # Noisy Image 50
    plt.subplot(4, 4, 9)
    plt.title('Noisy Image 50')
    plt.axis('off')
    plt.imshow(noisy_image_50, cmap='gray')

    # Masked Noisy Image 50
    plt.subplot(4, 4, 10)
    plt.title('Masked Noisy Image 50')
    plt.axis('off')
    plt.imshow(masked_noisy_50, cmap='gray')

    plt.tight_layout()
    plt.show()
    plt.waitforbuttonpress()
