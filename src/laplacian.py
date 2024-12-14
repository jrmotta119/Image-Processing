import numpy as np
import cv2
import matplotlib.pyplot as plt
import cmath

def fft(image, inverse=False):
    ##To recursively go through all N/2 sub matrices. 
    def fft_recursive(x):
        
        N = len(x)
        if N <= 1:
            return x
        even = fft_recursive(x[0::2])
        odd = fft_recursive(x[1::2])
        exp = 2j if inverse else -2j
        T = [cmath.exp(exp * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
        

        return [even[k] + T[k] for k in range(N // 2)] + \
               [even[k] - T[k] for k in range(N // 2)]
    if inverse:
        return np.fft.ifft2(image)
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float64) / 255.0 

    # rows = [image[i, :] for i in range(image.shape[0])] 

    # cols = [image[:, j] for j in range(image.shape[1])] 
    
    ## This is in case the dimensions are not power of 2
    ## Function taken from StackOverflow
    # new_rows = 2**int(np.ceil(np.log2(rows)))
    # new_cols = 2**int(np.ceil(np.log2(cols)))
    # padded_image = np.zeros((new_rows, new_cols), dtype=image.dtype)
    # padded_image[:rows, :cols] = image

    if not inverse:
        for x in range((image.shape[0])):
            for y in range(image.shape[0]):
                image[x, y] = image[x,y] * (-1) ** (x + y)

    
    row_transformed = [fft_recursive(row) for row in image]

    
    row_transformed = np.array(row_transformed).T  # Transpose to access columns
    col_transformed = [fft_recursive(col) for col in row_transformed]

    
    fft_result = np.array(col_transformed).T

    if inverse:
        N = image.shape[0]  
        return fft_result / (N**2)
    # fft_result = (fft_result + np.conj(np.flip(fft_result))) / 2
    return fft_result
    
def log(size, sigma):
   
    ## grid
    center = size // 2
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
    
    #formula
    norm = 1 / (2 * np.pi * sigma**4)
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    laplacian = (x**2 + y**2 - 2 * sigma**2) / sigma**2
    
    kernel = norm * laplacian * gaussian
    return kernel - kernel.mean()  # Normalize 


def apply(image, kernel):
    # For debugging reasons is in its separate function
    return cv2.filter2D(image, -1, kernel)

im = "../Images/cor_128.jpg" 
gim = cv2.imread(im, cv2.IMREAD_GRAYSCALE)


kSize = 18  ## Kernel size was suggested to round to nextodd integer (6*sigma)
sigma = 3.0      ## SD
kernel = log(kSize, sigma)

fourier = fft(gim)
filtered = apply(np.abs(fourier), kernel)

recontructed = fft(fourier,True)
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(gim, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Filtered Image")
plt.imshow(filtered, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Reconstructed Image")
plt.imshow(np.abs(recontructed), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()