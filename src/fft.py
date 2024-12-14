import cv2
import matplotlib.pyplot as plt
import numpy as np
import cmath
import time

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
    
    
if __name__ == "__main__":
    total_start = time.time()
    # image_size = 16
    # image = np.ones((image_size, image_size)) * 255  # White background
    # square_size = 7
    # start = (image_size - square_size) // 2
    # end = start + square_size
    # image[start:end, start:end] = 0  # Black square
    start = time.time()
    image = cv2.imread("../Images/cor_64.jpg",cv2.IMREAD_GRAYSCALE)
    print(type(image[0][0]))
    fourier = fft(image)
    end = time.time()
    print(f'fft time:{end-start:.6f} seconds')
    start = time.time()
    magnitude = np.abs(fourier)
    log_mag=(255/np.log10(255))*np.log10(1+(255/np.max(magnitude))*magnitude)
    log_mag2 = np.log(1+magnitude)


    plt.imshow(log_mag,cmap="gray")
    end = time.time()
    print(f'mag time:{end-start:.6f} seconds')
    total_end = time.time()
    print(f'total time:{total_end-total_start:.6f} seconds')

    plt.show()

    ## Inverse Test.
    fourier = fft(image, False)
    inverse_fourier = fft(fourier, True)
    magnitude = np.abs(inverse_fourier)
    # log_mag=(255/np.log10(255))*np.log10(1+(255/np.max(magnitude))*magnitude)

    # inverse_real = np.real(inverse_fourier)
    normalized_dct = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # output_image = np.uint8(normalized_dct)
    cv2.imwrite("FFT_BL_64.jpg",normalized_dct)

