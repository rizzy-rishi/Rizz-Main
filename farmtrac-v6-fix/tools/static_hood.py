# Static-plate fix: the tractor is locked in this shot, so replace the hood with one clean plate
# (temporal median of person-free pixels) carrying the badges, glued at a fixed position.
import cv2, numpy as np, glob, os, sys
S = sys.argv[1]; REF = 156
SC = 4; X0, Y0, W, H = 680, 420, 340, 150
BX0, BY0, BX1, BY1 = 420, 380, 1040, 670          # working box around the hood
frames = sorted(glob.glob(f"{S}/wide/frames/*.png")); ns = [int(os.path.basename(f)[:4]) for f in frames]
stack = np.zeros((len(ns), BY1-BY0, BX1-BX0, 3), np.uint8); valid = np.zeros(stack.shape[:3], bool)
for i, n in enumerate(ns):
    stack[i] = cv2.imread(frames[i])[BY0:BY1, BX0:BX1]
    pm = cv2.imread(f"{S}/wide/mask/{n:04d}.png", 0)[BY0:BY1, BX0:BX1]
    valid[i] = cv2.dilate((pm > 40).astype(np.uint8), np.ones((25, 25), np.uint8)) == 0
plate = np.zeros(stack.shape[1:], np.float32)
for r in range(0, stack.shape[1], 40):              # masked median in row strips
    s = stack[:, r:r+40].astype(np.float32); v = valid[:, r:r+40]
    s[~v] = np.nan; plate[r:r+40] = np.nanmedian(s, 0)
holes = np.isnan(plate).any(2)
plate = np.nan_to_num(plate); print("hole px", int(holes.sum()))
ref = cv2.imread(f"{S}/wide/frames/{REF:04d}.png").astype(np.float32)
plate[holes] = ref[BY0:BY1, BX0:BX1][holes]
# badges + corrected stripe, fixed position (identity; tractor does not move)
layer = cv2.GaussianBlur(np.load(f"{S}/wide/layer.npy").astype(np.float32), (0, 0), 1.3)
small = cv2.resize(layer, (W, H), interpolation=cv2.INTER_AREA); a = small[..., 3:]/255
ys, xs = Y0-BY0, X0-BX0
plate[ys:ys+H, xs:xs+W] = plate[ys:ys+H, xs:xs+W]*(1-a) + small[..., :3]*a
# hood region mask, kept inside the body, feathered
poly = np.array([[446, 402], [700, 412], [900, 440], [1004, 470], [1010, 612], [700, 618], [446, 652]]) - [BX0, BY0]
hm = np.zeros(plate.shape[:2], np.float32); cv2.fillPoly(hm, [poly.astype(np.int32)], 1)
hm = cv2.GaussianBlur(cv2.erode(hm, np.ones((5, 5), np.uint8)), (0, 0), 4)
cv2.imwrite(f"{S}/wide/plate.png", plate.astype(np.uint8))
# sparkle ghost
spark = np.zeros((1080, 1920), np.uint8); t = np.linspace(0, 2*np.pi, 400); cx, cy, r = 1740, 893, 38
cv2.fillPoly(spark, [np.stack([cx + r*np.sign(np.cos(t))*np.abs(np.cos(t))**3, cy + r*np.sign(np.sin(t))*np.abs(np.sin(t))**3], 1).astype(np.int32)], 255)
spark = cv2.dilate(spark, np.ones((7, 7), np.uint8))
ssoft = cv2.GaussianBlur(cv2.dilate(spark, np.ones((15, 15), np.uint8)).astype(np.float32)/255, (0, 0), 6)[..., None]
gains = []
for i, n in enumerate(ns):
    fr = cv2.imread(frames[i]); fr = cv2.inpaint(fr, spark, 12, cv2.INPAINT_NS).astype(np.float32)
    fr = fr*(1-ssoft) + cv2.GaussianBlur(fr, (0, 0), 5)*ssoft
    box = fr[BY0:BY1, BX0:BX1]
    pm = cv2.imread(f"{S}/wide/mask/{n:04d}.png", 0)[BY0:BY1, BX0:BX1].astype(np.float32)/255
    occ = cv2.GaussianBlur(cv2.dilate((pm > 0.3).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 2)
    sel = (hm > 0.9) & (occ < 0.02) & (plate.mean(2) > 30) & (plate.mean(2) < 140) & (box.mean(2) > 30)
    g = np.clip(np.median(box[sel].mean(1)) / (np.median(plate[sel].mean(1)) + 1e-3), 0.92, 1.08) if sel.sum() > 500 else 1.0
    gains.append(g)
    a = (hm*(1-occ))[..., None]
    fr[BY0:BY1, BX0:BX1] = box*(1-a) + (plate*g)*a
    cv2.imwrite(f"{S}/wide/out2/{n:04d}.png", np.clip(fr, 0, 255).astype(np.uint8))
g = np.array(gains); print("gain range", g.min().round(3), g.max().round(3))
