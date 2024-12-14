import cv2
import matplotlib.pyplot as plt
import numpy as np
import cmath
import time

def fft(image, inverse=False):
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
## Filter developed with parts of a Stack Overflow post
def ideal(N, D0, type='low'):
    
    x, y = N // 2, N // 2
    mask = np.zeros((N, N))

    for u in range(N):
        for v in range(N):
            D = np.sqrt((u - x)**2 + (v - y)**2)
            if type == 'low' and D <= D0:
                mask[u, v] = 1
            elif type == 'high' and D > D0:
                mask[u, v] = 1

    return mask

def butterworth(N, D0, n, type='low'):
    
    x, y = N // 2, N // 2
    mask = np.zeros((N, N))

    for u in range(N):
        for v in range(N):
            D = np.sqrt((u - x)**2 + (v - y)**2)
            if type == 'low':
                mask[u, v] = 1 / (1 + (D / D0)**(2 * n))
            elif type == 'high':
                mask[u, v] = 1 / (1 + (D0 / D)**(2 * n)) if D != 0 else 0

    return mask

def applyFilter(image, mask):
    fourier = fft(image, False)
   
    fftRes = fourier * mask

    filtered = fft(fftRes,True)
    
    return np.abs(filtered)


if __name__ == "__main__":
    startTotal = time.time()
    # image_size = 16
    # image = np.zeros((image_size, image_size))  # White background
    # square_size = 7
    # start = (image_size - square_size) // 2
    # end = start + square_size
    # image[start:end, start:end] = 255  # Black square
    image = cv2.imread("../Images/decor_128.jpg",cv2.IMREAD_GRAYSCALE)
    
    
    D0 = 50  
    n = 10    

    
    start = time.time()
    ideal_low = ideal(image.shape[0], D0, type='low')
    end = time.time()
    print(f'ideal_low time:{end-start:.6f} seconds')
    start = time.time()
    ideal_high = ideal(image.shape[0], D0, type='high')
    end = time.time()
    print(f'ideal_high time:{end-start:.6f} seconds')

   
    start = time.time()
    bLow = butterworth(image.shape[0], D0, n, type='low')
    end = time.time()
    print(f'bLow time:{end-start:.6f} seconds')
    start = time.time()
    bHigh = butterworth(image.shape[0], D0, n, type='high')
    end = time.time()
    print(f'bHigh time:{end-start:.6f} seconds')

    # Apply Filters
    idealLow = applyFilter(image, ideal_low)
    idealHigh = applyFilter(image, ideal_high)
    bLowFiltered = applyFilter(image, bLow)
    bHighFiltered = applyFilter(image, bHigh)


   
    
    # start = time.time()
    
    # log_mag=(255/np.log10(255))*np.log10(1+(255/np.max(magnitude))*magnitude)
    # log_mag2 = np.log(1+magnitude)


    # fft_result_numpy = np.fft.fft2(image)
    # ifft_result_numpy = np.fft.ifft2(fft_result_numpy)

    # plt.imshow(log_mag,cmap="gray")
    # end = time.time()
    # print(f'mag time:{end-start:.6f} seconds')
    # inverse = fft(fourier,True)

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1),plt.imshow(np.abs(image),cmap='gray'), plt.title("Original")
    plt.subplot(3, 2, 2),plt.imshow(np.abs(idealLow),cmap='gray'),plt.title("Ideal Low")
    plt.subplot(3, 2, 3),plt.imshow(np.abs(idealHigh),cmap='gray'),plt.title("Ideal High")
    plt.subplot(3, 2, 4),plt.imshow(np.abs(bHighFiltered),cmap='gray'),plt.title("Butterworth High")
    plt.subplot(3, 2, 5),plt.imshow(np.abs(bLowFiltered),cmap='gray'),plt.title("Butterworth Low")
    plt.tight_layout()

    plt.show()
    endTotal = time.time()
    print(f'total time:{endTotal-startTotal:.6f} seconds')
    # print("Original Image:\n", image)
    # print("\nFFT Result (Magnitude):\n", np.abs(fourier))
    # print("NumPy FFT Result (Magnitude):\n", np.abs(fft_result_numpy))
    # print("\nReconstructed Image:\n", np.real(inverse))
    # print("NumPy Reconstructed Image:\n", np.abs(ifft_result_numpy))    
