from typing import List

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
    """Compute the Mean Squared Error between two images."""
    err = np.sum((original_image.astype("float") - distorted_image.astype("float")) ** 2)
    err /= float(original_image.shape[0] * original_image.shape[1])
    return err

def psnr(image1, image2, max_pixel_value=255.0):
    """Compute the Peak Signal-to-Noise Ratio between two images."""
    mean_squared_err = mse(image1, image2)
    if mean_squared_err == 0:
        return "Same Image"
    return 20 * np.log10(max_pixel_value / np.sqrt(mean_squared_err))

image = cv2.imread('lena.tif', cv2.IMREAD_GRAYSCALE)

noisy_image = add_gaussian_noise(image, mean=0, std_dev=50)

original_histogram = calculate_histogram(image)
noisy_histogram = calculate_histogram(noisy_image)

plt.figure(figsize=(8, 12))

plt.subplot(4, 2, 1)
plt.title('Original Image')
plt.axis('off')
plt.imshow(image, cmap='gray')

plt.subplot(4, 2, 2)
plt.title('Histogram (Original)')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.bar(range(256), original_histogram)

plt.subplot(4, 2, 3)
plt.title('Noisy Image')
plt.axis('off')
plt.imshow(noisy_image, cmap='gray')

plt.subplot(4, 2, 4)
plt.title('Histogram (Noisy)')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.bar(range(256), noisy_histogram)

plt.tight_layout()
plt.show()
