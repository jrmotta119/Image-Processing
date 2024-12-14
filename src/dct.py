import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def dct(image):
    N = image.shape[0]
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
                    sigma += image[x,y] * math.cos((2*x + 1) * u *math.pi/(2*N)) * math.cos((2*y + 1) * v *math.pi/(2*N))
            dct_im[u,v]=tau * sigma
    return dct_im
    # mag = np.abs(dct_im)
    # return np.log(1+mag)
def idct(dct_coefficients):
    N = dct_coefficients.shape[0]
    reconstructed = np.zeros((N, N))
    for x in range(N):
        for y in range(N):
            sigma = 0
            for u in range(N):
                for v in range(N):
                    tau = (1 / N) if (u == 0 and v == 0) else (2 / N)
                    sigma += tau * dct_coefficients[u, v] * math.cos((2 * x + 1) * u * math.pi / (2 * N)) * math.cos((2 * y + 1) * v * math.pi / (2 * N))
            reconstructed[x, y] = sigma
    return reconstructed

def calc_product(n,bu,bv,bx,by):
    prod = 1
    for i in range(n):
        pow = (int(bx[i]) * int(bu[n-1-i])) + (int(by[i])* int(bv[n-1-i]))
        prod *= (-1)**(pow)
    return prod

# def walshVersion2(image):
#     N = image.shape[0]
#     n=int(math.log2(N))
#     walsh_im=np.zeros((N,N))
#     walsh_matrix = np.empty((N,N))
#     for u in range(N):
#         for v in range(N):
#             sigma=0
#             bu = bin(u)[2:].zfill(n)
#             bv = bin(v)[2:].zfill(n)
#             for x in range(N):
#                 bx = bin(x)[2:].zfill(n)
#                 for y in range(N):
#                     by = bin(y)[2:].zfill(n)
#                     product = calc_product(n,bu,bv,bx,by)
#                     sigma += image[x,y] * product
#                     walsh_matrix[x,y]=product
#             walsh_im[u,v]= sigma/N
    
#     mag = np.abs(walsh_im)
#     return np.log1p(mag)
    
def walshBit(n, k):
    
    n = np.array([int(b) for b in format(n, 'b').zfill(k.bit_length())]) 
    k = np.array([int(b) for b in format(k, 'b').zfill(k.bit_length())])

    #to keep same size bits
    max_len = max(len(n), len(k))
    n = np.pad(n, (max_len - len(n), 0), constant_values=0) 
    k = np.pad(k, (max_len - len(k), 0), constant_values=0)
    return (-1) ** np.sum(n & k) #bitwise operation

def walsh(image):
    
    N = image.shape[0]
    walsh_transform = np.zeros_like(image, dtype=float)
    
    for u in range(N):
        for v in range(N):
            coeff_sum = 0
            for x in range(N):
                for y in range(N):
                    w_x = walsh(x, u)
                    w_y = walsh(y, v)
                    coeff_sum += image[x, y] * w_x * w_y
            walsh_transform[u, v] = coeff_sum / N  
    return walsh_transform

def matrix(N):
    if N == 1:
        return np.array([[1]])
    else:
        ## Sub matrices recursively
        H_prev = matrix(N // 2)
        ## Tried coding manually the whole HA(r,m,x) but was too slow.
        ## This stacks the lower matrices
        H1 = np.kron(H_prev, [1, 1])  
        ## multiplyting previous matrices with Normalized eye should do the trick
        H2 = np.kron(np.eye(N // 2), [1, -1]) * np.sqrt(2)
        H = np.vstack((H1, H2))
        return H / np.sqrt(2)  # Normalize the entire matrix





def haar(image):
    N = image.shape[0]
    H=matrix(N)
    return np.dot(H, np.dot(image,H.T))
    

 

if __name__ == "__main__":

    # image_size = 16
    # image = np.ones((image_size, image_size)) * 255  # White background
    # square_size = 7
    # start = (image_size - square_size) // 2
    # end = start + square_size
    # image[start:end, start:end] = 0  # Black square

    # cv2.imwrite("BL_16.jpg",image)
    im=cv2.imread('../Images/decor_64.jpg')
    image = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    #dct
    dct_forward = dct(image)
    rImage = idct(dct_forward)
    normalized_dct = cv2.normalize(rImage, None, 0, 255, cv2.NORM_MINMAX)

    output_image = np.uint8(normalized_dct)

    cv2.imwrite("DCT_Decor_64.jpg",output_image)
    plt.imshow(output_image,cmap="gray")
    # plt.savefig("DCT_Cor_32")
    plt.show() 
    # Inverse = walsh(tImage)
    # plt.imshow(Inverse,cmap="gray")
    # plt.show()