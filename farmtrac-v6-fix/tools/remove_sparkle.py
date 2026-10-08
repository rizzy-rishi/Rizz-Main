import cv2, numpy as np, sys
src, out = sys.argv[1], sys.argv[2]
img = cv2.imread(src)
mask = np.zeros(img.shape[:2], np.uint8)
def star(cx, cy, r):
    # 4-point concave sparkle (astroid-like), slightly oversized
    t = np.linspace(0, 2*np.pi, 400)
    x = cx + r*np.sign(np.cos(t))*np.abs(np.cos(t))**3
    y = cy + r*np.sign(np.sin(t))*np.abs(np.sin(t))**3
    cv2.fillPoly(mask, [np.stack([x, y], 1).astype(np.int32)], 255)
star(int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
star(int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]))
mask = cv2.dilate(mask, np.ones((7,7), np.uint8))
cv2.imwrite(out.replace('.png','_mask.png'), mask)
res = cv2.inpaint(img, mask, 12, cv2.INPAINT_NS)
cv2.imwrite(out, res)
