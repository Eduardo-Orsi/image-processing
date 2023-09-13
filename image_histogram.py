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
    """Compute the Mean Squared Error between two images."""
    err = np.sum((original_image.astype("float") - distorted_image.astype("float")) ** 2)
    err /= float(original_image.shape[0] * original_image.shape[1])
    return err

def psnr(original_image, distorted_image, max_pixel_value=255.0) -> Union[float, str]:
    """Compute the Peak Signal-to-Noise Ratio between two images."""
    mean_squared_err = mse(original_image, distorted_image)
    if mean_squared_err == 0:
        return "Same Image"
    return 20 * np.log10(max_pixel_value / np.sqrt(mean_squared_err))

def modified_psnr(original_image, distorted_image, max_pixel_value=255.0) -> Union[float, str]:
    """Compute the modified PSNR between two images."""
    mean_squared_err = mse(original_image, distorted_image)
    if mean_squared_err == 0:
        return "Same Image"
    return 10 * np.log10(max_pixel_value / mean_squared_err)

def apply_mask(image: NDArray, mask: NDArray) -> NDArray:
    output_image = np.zeros_like(image)
    image_height, image_width = image.shape
    mask_height, mask_width = mask.shape
    mask_center_x = mask_width // 2
    mask_center_y = mask_height // 2

    for y in range(mask_center_y, image_height - mask_center_y):
        for x in range(mask_center_x, image_width - mask_center_x):
            roi = image[y - mask_center_y:y + mask_center_y + 1, x - mask_center_x:x + mask_center_x + 1]
            convolution_result = np.sum(roi * mask)
            output_image[y, x] = convolution_result

    return output_image


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
