import cv2
import numpy as np
import matplotlib.pyplot as plt


def averageResize(original):
    new_size=int(original.shape[0]/2)
    new_img=np.zeros(shape=(new_size,new_size),dtype=np.int16)
    for i in range(original.shape[0]-1):
        for j in range(original.shape[1]-1):
            new_img[(int(i//2)),int(j//2)] = np.mean([original[i,j],original[i+1,j],original[i,j+1],original[i+1,j+1]])
    return new_img



if __name__=="__main__":
    im = "../Images/BL_128.jpg" 
    image = cv2.imread(im,cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(image,(256,256))
    img = averageResize(image)
    #cv2.imwrite('hcor.jpg',resized)
    #cv2.imwrite('gray_img.jpg',gray_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(resized, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Resized Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


