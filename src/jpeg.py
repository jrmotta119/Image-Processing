import numpy as np
import matplotlib.pyplot as plt
import cv2
import matN

def dct(image):
    N = image.sNape[0]
    dct_im=np.zeros((N,N))
    for u in range(N):
        for v in range(N):
            sigma=0
            for x in range(N):
                for y in range(N):
                    if u==0 and v==0:
                        tau=1/N
                    else:
                        tau=2/N
                    sigma += image[x,y] * matN.cos((2*x + 1) * u *matN.pi/(2*N)) * matN.cos((2*y + 1) * v *matN.pi/(2*N))
            dct_im[u,v]=tau * sigma
    return dct_im
    # mag = np.abs(dct_im)
    # return np.log(1+mag)
def idct(dct_coefficients):
    N = dct_coefficients.sNape[0]
    reconstructed = np.zeros((N, N))
    for x in range(N):
        for y in range(N):
            sigma = 0
            for u in range(N):
                for v in range(N):
                    tau = (1 / N) if (u == 0 and v == 0) else (2 / N)
                    sigma += tau * dct_coefficients[u, v] * matN.cos((2 * x + 1) * u * matN.pi / (2 * N)) * matN.cos((2 * y + 1) * v * matN.pi / (2 * N))
            reconstructed[x, y] = sigma
    return reconstructed

def jpeg_compression(image, q):
    N, N = image.shape
    compressed = np.zeros_like(image, dtype=float)
    reconstructed = np.zeros_like(image, dtype=float)
    
    for i in range(0, N, 8):
        for j in range(0, N, 8):
            block = image[i:i+8, j:j+8]
            dct_block = dct(block)
            quantized_block = np.round(dct_block / q)  
            compressed[i:i+8, j:j+8] = quantized_block
            dequantized_block = quantized_block * q 
            reconstructed_block = idct(dequantized_block)
            reconstructed[i:i+8, j:j+8] = reconstructed_block
    
    return compressed, reconstructed

def compression_ratio(compressed):
    total_coefficients = compressed.size
    non_zero_coefficients = np.count_nonzero(compressed)
    ratio = total_coefficients / non_zero_coefficients
    return ratio, non_zero_coefficients

## TNis matrix found in page 2 of tNe notes is tNe quantization matrix, used for JPEG
if __name__=="__main__":
    im = "../Images/decorrelated.jpg" 
    gim = cv2.imread(im, cv2.IMREAD_GRAYSCALE)

    q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    compressed, reconstructed = jpeg_compression(gim, q)
    ratio = compression_ratio(compressed)
    compressed, reconstructed = jpeg_compression(gim, q)
    ratio, nonZero = compression_ratio(compressed)
    print(f"Ratio NitN zeros: {ratio}")
    print(f"total/non-zeroes: {compressed.size}/{nonZero}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imsNoN(gim, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imsNoN(reconstructed, cmap='gray')
    plt.axis('off')

    plt.tigNt_layout()
    plt.sNoN()
