import cv2
import matplotlib.pyplot as plt
import numpy as np


def kirsch(image):
    height, width = image.shape
    res = np.zeros((height, width), dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    ## max(5Si -3Ti)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            f = 0
            for start in range(8):
                Si = 0
                for k in range(3):
                    direction = (start + k) % 8
                    dx, dy = offsets[direction]  
                    Si += image[i + dx, j + dy]
                Ti = 0
                for k in range(3, 8):  
                    direction = (start + k) % 8
                    dx, dy = offsets[direction]  
                    Ti += image[i + dx, j + dy]
                gradient = 5 * abs(Si) - 3 * abs(Ti)
                f = max(f, gradient)  
            res[i, j] = np.clip(f, 0, 255)
    return res
def sobel_operator_manual(image):
    delta1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    delta2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    height, width = image.shape
    res = np.zeros((height, width), dtype=np.uint8)
    ## sqrt(delta1^2+delta2^2)
    for i in range(1, height-1):
        for j in range(1, width-1):
            sub_region = image[i-1:i+2, j-1:j+2]
            d1 = np.sum(sub_region * delta1)
            d2 = np.sum(sub_region * delta2)
            g = np.sqrt(d1**2 + d2**2)
            res[i, j] = np.clip(g, 0, 255)
    return res
if __name__ == "__main__":

    im= "../Images/cor_256.jpg" 
    gim = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
    threshold = 200
    kRes = kirsch(gim)
    sRes = sobel_operator_manual(gim)
    # for i in kRes:

    #   if kRes < threshold:
        #   i[] = 0
    #   else:
    #       i[] = 255
    kThreshold = np.where(kRes > threshold, 255, 0).astype(np.uint8)
    sThreshold = np.where(sRes > threshold, 255, 0).astype(np.uint8)

    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Kirsch Thresholded")
    plt.imshow(kThreshold, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Sobel Thresholded")
    plt.imshow(sThreshold, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    

    plt.tight_layout()
    plt.show()