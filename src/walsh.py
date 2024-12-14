import numpy as np
import matplotlib.pyplot as plt

def walsh_basis(n, k):
    
    binary_n = np.array([int(b) for b in format(n, 'b').zfill(k.bit_length())])
    binary_k = np.array([int(b) for b in format(k, 'b').zfill(k.bit_length())])

    max_len = max(len(binary_n), len(binary_k))
    binary_n = np.pad(binary_n, (max_len - len(binary_n), 0), constant_values=0)
    binary_k = np.pad(binary_k, (max_len - len(binary_k), 0), constant_values=0)
    return (-1) ** np.sum(binary_n & binary_k)

def walsh_transform_2d(image):
    
    N = image.shape[0]
    walsh_transform = np.zeros_like(image, dtype=float)
    
    for u in range(N):
        for v in range(N):
            # Compute W(u, v) using the Walsh Transform equation
            coeff_sum = 0
            for x in range(N):
                for y in range(N):
                    w_x = walsh_basis(x, u)
                    w_y = walsh_basis(y, v)
                    coeff_sum += image[x, y] * w_x * w_y
            walsh_transform[u, v] = coeff_sum / N  # Normalize by N
    return walsh_transform

# Create a black square image with white background
image_size = 8  # Image dimensions must be a power of 2
image = np.ones((image_size, image_size)) * 255  # White background
square_size = 4
start = (image_size - square_size) // 2
end = start + square_size
image[start:end, start:end] = 0  # Black square in the center

# Compute the Walsh Transform
walsh_result = walsh_transform_2d(image)

# Visualize the original image and its Walsh Transform
plt.figure(figsize=(12, 6))

walsh_inverse=walsh_transform_2d(walsh_result)

# Original image
plt.subplot(1, 2, 1)
plt.imshow(walsh_inverse, cmap='gray')
plt.title("Original Image (Black Square)")
plt.colorbar()

# Walsh Transform result
plt.subplot(1, 2, 2)
plt.imshow(np.log1p(np.abs(walsh_result)), cmap='gray')  # Log scale for visibility
plt.title("Walsh Transform")
plt.colorbar()

plt.show()