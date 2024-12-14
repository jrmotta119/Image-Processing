import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('../Images/BL_128.jpg')
resized = cv2.resize(image,(512,512))
gray_image=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)

(row, col) = gray_image.shape[0:2]
negative = np.zeros(shape=(row,col))

for i in range(row):
    for j in range(col):
        negative[i,j] = 255-gray_image[i,j]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Negative Image")
plt.imshow(negative, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
