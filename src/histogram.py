import cv2
import numpy as np
import matplotlib.pyplot as plt

def dynamic_range_expansion(image, a, b, rl=0, rk=255):
    
    ## r' = (rk-r1)/(b-a) * (r-a) + r1
    expanded_image = ((image - a) / (b - a) * (rk - rl) + rl).clip(rl, rk).astype(np.uint8)
    return expanded_image

def histogram_equalization(image):
    ## Convert to histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    ## PDF
    pdf = hist / np.sum(hist)
    ## SIGMA Pr(r)
    cdf = pdf.cumsum()
    ## Normalize (c - min/max - min)
    cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255
    cdf_normalized = cdf_normalized.astype(np.uint8)
    equalized_image = cdf_normalized[image]     ## Map

    
    return equalized_image

im= "../Images/cor_256.jpg" 
gim = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
a, b = gim.min(), gim.max()
expanded = dynamic_range_expansion(gim, a, b)
equalized = histogram_equalization(gim)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(gim, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Dynamic Range Expanded")
plt.imshow(expanded, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Equalized Image")
plt.imshow(equalized, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()