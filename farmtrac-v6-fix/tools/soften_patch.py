import cv2, numpy as np, sys
img = cv2.imread(sys.argv[1]).astype(np.float32)
m = cv2.imread(sys.argv[2], 0).astype(np.float32)/255
m = cv2.GaussianBlur(cv2.dilate(m, np.ones((15,15))), (0,0), 6)[...,None]
blur = cv2.GaussianBlur(img, (0,0), 5)
cv2.imwrite(sys.argv[3], (img*(1-m)+blur*m).clip(0,255).astype(np.uint8))
