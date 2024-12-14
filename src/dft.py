import cv2
import numpy as np
import matplotlib.pyplot as plt
import cmath


def createSquare(image_size,square_size):
    # image = cv2.imread('../Images/gray_img.jpg')
    new_image = np.ones((image_size,image_size))*255
    
    start = (image_size -square_size)//2
    end = start + square_size
    new_image[start:end,start:end] = 0
            
    return new_image

def averageResize(original):
    new_size=int(original.shape[0]/2)
    new_img=np.zeros(shape=(new_size,new_size),dtype=np.int16)
    for i in range(original.shape[0]-1):
        for j in range(original.shape[1]-1):
            new_img[(int(i//2)),int(j//2)] = np.mean([original[i,j],original[i+1,j],original[i,j+1],original[i+1,j+1]])
    return new_img

def dft(image):
    N = image.shape[0]
    dft=np.zeros((N,N),dtype=complex)
    for u in range(N):
        for v in range(N):
            sigma=0 +0j
            for x in range(N):
                for y in range(N):
                    exponent = -2j * cmath.pi * (((u*x/N)+(v*y/N)))
                    sigma += image[x,y]*((-1)**(x+y)) * cmath.exp(exponent)
                    #print(sigma)
            dft[u,v]=sigma/(N**2)
    #dft_shift = np.fft.fftshift(dft)
    #mSpectrum=np.log(np.abs(dft)+1)
    return np.log(((255/np.abs(np.max(dft)))*np.abs(dft))+1)

if __name__=="__main__":
    size = 8
    square_size = 4
    image = createSquare(size,square_size)
    # im=cv2.imread('../Images/BL_32.jpg')
    # gray_image = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # image= averageResize(gray_image)
    # cv2.imwrite("BL_8.jpg",image)

    mSpectrum = dft(image)

    plt.imshow(mSpectrum,cmap="gray")
    plt.show()
    #cv2.imshow("dft",mSpectrum)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()     
