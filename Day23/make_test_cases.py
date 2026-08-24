import cv2
import numpy as np

frame = cv2.imread("Input/Car.jpg")

# 1. Angled — rotate the image
h, w = frame.shape[:2]
M = cv2.getRotationMatrix2D((w // 2, h // 2), 25, 1.0)  # 25 degree tilt
angled = cv2.warpAffine(frame, M, (w, h))
cv2.imwrite("Input/angled.jpg", angled)

# 2. Blurry — simulate motion blur
kernel_size = 25
kernel = np.zeros((kernel_size, kernel_size))
kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
kernel = kernel / kernel_size
blurry = cv2.filter2D(frame, -1, kernel)
cv2.imwrite("Input/blurry.jpg", blurry)

# 3. Low-res — shrink then upscale (simulates a far-away vehicle)
small = cv2.resize(frame, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_LINEAR)
lowres = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
cv2.imwrite("Input/lowres.jpg", lowres)

print("Created angled.jpg, blurry.jpg, lowres.jpg in Input/")