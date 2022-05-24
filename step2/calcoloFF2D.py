import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

image = plt.imread('/Users/letiziagirardi/Desktop/THESIS/HPFCN/HPFCN/output_frame/OPN/postProcessed/boat/0000.png')
plt.figure

# original image
plt.subplot(121)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.show()

# fft2d
img_complex = tf.cast(image, dtype=tf.complex64)
img_T = tf.transpose(img_complex)
fft_imgT = tf.signal.fft2d(img_T)
fft_img = tf.transpose(fft_imgT)

# plot fft2
real = tf.math.real(fft_img)
plt.imshow(real.numpy())
plt.title('Real part of Fft2')
plt.show()

# maschera
total_rows, total_cols, total_layers = fft_img.shape
print(fft_img.shape)
X, Y, Z = np.ogrid[:total_rows, :total_cols, :total_layers]

circleRadius = 50
center_row, center_col = total_rows / 2, total_cols / 2
dist_from_center = (X - center_row) ** 2 + (Y - center_col) ** 2

circular_mask = (dist_from_center <= circleRadius ** 2)

z_masked = tf.multiply(real, circular_mask)
plt.imshow(tf.math.real(z_masked))
plt.title('Masked of real part of image')
plt.show()

# ifft2
imaginary = tf.math.imag(fft_img)
imfft_filtered = tf.multiply(fft_img, circular_mask)
imfft_filtered *= 255
masked_T = tf.transpose(imfft_filtered)  # trasposta
filtered_im1 = tf.signal.ifft2d(masked_T)
filtered_im1_T = tf.transpose(filtered_im1)
plt.imshow(tf.math.real(filtered_im1_T).numpy())
plt.title('Real part of ifft2')
plt.show()

# x calcolare la varianza
print(tf.math.reduce_variance(filtered_im1_T, axis=(0, 1)))
