import cv2
import matplotlib.pyplot as plt
import numpy as np
import cmath


def mse(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    diff = (img1 - img2) ** 2
    
    mse = np.mean(diff)
    
    return mse
if __name__ == "__main__":
    image = cv2.imread("../Images/Diff/decor_64.jpg",cv2.IMREAD_GRAYSCALE)
    dct = cv2.imread("../Images/Diff/DCT_Decor_64.jpg",cv2.IMREAD_GRAYSCALE)
    fft = cv2.imread("../Images/Diff/FFT_Decor_64.jpg",cv2.IMREAD_GRAYSCALE)
    
    mse_DCT = mse(image,dct)
    mse_FFT = mse(image,fft)

    print("MSE of DCT: " + str(mse_DCT))
    print("MSE of FFT: " + str(mse_FFT))
