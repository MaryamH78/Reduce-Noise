import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# بارگذاری تصویر
image = cv2.imread(r'C:\Users\Mehr\Desktop\Picture\nature.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# تابع برای افزودن نویز گاوسی
def add_gaussian_noise(image, mean=0, var=0.01):
    sigma = var ** 0.5
    gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) / 255.0 + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 1) * 255
    return noisy_image.astype(np.uint8)

# تابع برای افزودن نویز اسپکل
def add_speckle_noise(image, mean=0, var=0.04):
    gaussian_noise = np.random.normal(mean, var ** 0.5, image.shape).astype(np.float32)
    speckle_noise = image.astype(np.float32) / 255.0 + image.astype(np.float32) / 255.0 * gaussian_noise
    speckle_noise = np.clip(speckle_noise, 0, 1) * 255
    return speckle_noise.astype(np.uint8)

# افزودن نویز گاوسی به تصویر
gaussian_noisy_image = add_gaussian_noise(image_rgb)

# افزودن نویز اسپکل به تصویر
speckle_noisy_image = add_speckle_noise(image_rgb)

# حذف نویز گاوسی با فیلتر گاوسی
gaussian_denoised_image = cv2.GaussianBlur(gaussian_noisy_image, (5, 5), 1.5)

# حذف نویز اسپکل با فیلتر میانه
speckle_denoised_image_median = np.zeros_like(speckle_noisy_image)
for i in range(3):
    speckle_denoised_image_median[:, :, i] = cv2.medianBlur(speckle_noisy_image[:, :, i], 3)

# حذف نویز اسپکل با فیلتر ویولت
def wavelet_denoise_channel(channel, wavelet='db1', level=1):
    coeffs = pywt.wavedec2(channel, wavelet, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    denoised_channel = pywt.waverec2(coeffs_H, wavelet)
    return np.clip(denoised_channel, 0, 255).astype(np.uint8)

speckle_denoised_image_wavelet = np.zeros_like(speckle_noisy_image)
for i in range(3):
    speckle_denoised_image_wavelet[:, :, i] = wavelet_denoise_channel(speckle_noisy_image[:, :, i], wavelet='db1', level=1)

# نمایش تصاویر
plt.figure(figsize=(12, 8))

# تصویر اصلی
plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis('off')

# تصویر با نویز گاوسی
plt.subplot(2, 3, 2)
plt.title("Gaussian Noisy Image")
plt.imshow(gaussian_noisy_image)
plt.axis('off')

# تصویر پس از حذف نویز گاوسی
plt.subplot(2, 3, 3)
plt.title("Denoised Image (Gaussian)")
plt.imshow(gaussian_denoised_image)
plt.axis('off')

# تصویر با نویز اسپکل
plt.subplot(2, 3, 4)
plt.title("Speckle Noisy Image")
plt.imshow(speckle_noisy_image)
plt.axis('off')

# تصویر پس از حذف نویز اسپکل با فیلتر میانه
plt.subplot(2, 3, 5)
plt.title("Denoised Image (Median)")
plt.imshow(speckle_denoised_image_median)
plt.axis('off')

# تصویر پس از حذف نویز اسپکل با فیلتر ویولت
plt.subplot(2, 3, 6)
plt.title("Denoised Image (Wavelet)")
plt.imshow(speckle_denoised_image_wavelet)
plt.axis('off')

plt.show()

# ذخیره تصاویر (در صورت نیاز)
# cv2.imwrite('gaussian_noisy_image.jpg', cv2.cvtColor(gaussian_noisy_image, cv2.COLOR_RGB2BGR))
# cv2.imwrite('gaussian_denoised_image.jpg', cv2.cvtColor(gaussian_denoised_image, cv2.COLOR_RGB2BGR))
# cv2.imwrite('speckle_noisy_image.jpg', cv2.cvtColor(speckle_noisy_image, cv2.COLOR_RGB2BGR))
# cv2.imwrite('speckle_denoised_image_median.jpg', cv2.cvtColor(speckle_denoised_image_median, cv2.COLOR_RGB2BGR))
# cv2.imwrite('speckle_denoised_image_wavelet.jpg', cv2.cvtColor(speckle_denoised_image_wavelet, cv2.COLOR_RGB2BGR))
