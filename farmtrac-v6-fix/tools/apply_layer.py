# Composite the tracked decal layer on every wide-shot frame, occluded by the person matte.
import cv2, numpy as np, json, glob, os, sys
S = sys.argv[1]; REF = int(sys.argv[2])
SC = 4; X0, Y0, W, H = 680, 420, 340, 150
layer = np.load(f"{S}/wide/layer.npy").astype(np.float32)
layer = cv2.GaussianBlur(layer, (0, 0), 1.3)                       # prefilter before down-warp
Hs = {int(k): np.array(v) for k, v in json.load(open(f"{S}/wide/H.json")).items()}
ref = cv2.imread(f"{S}/wide/frames/{REF:04d}.png").astype(np.float32)
T = np.array([[1/SC, 0, X0], [0, 1/SC, Y0], [0, 0, 1]])            # canvas -> ref coords
# sparkle ghost (screen-fixed)
spark = np.zeros((1080, 1920), np.uint8); t = np.linspace(0, 2*np.pi, 400); cx, cy, r = 1740, 893, 38
cv2.fillPoly(spark, [np.stack([cx + r*np.sign(np.cos(t))*np.abs(np.cos(t))**3, cy + r*np.sign(np.sin(t))*np.abs(np.sin(t))**3], 1).astype(np.int32)], 255)
spark = cv2.dilate(spark, np.ones((7, 7), np.uint8))
ssoft = cv2.GaussianBlur(cv2.dilate(spark, np.ones((15, 15), np.uint8)).astype(np.float32)/255, (0, 0), 6)[..., None]
ref_alpha = (layer[..., 3] > 128).astype(np.uint8)
for f in sorted(glob.glob(f"{S}/wide/frames/*.png")):
    n = int(os.path.basename(f)[:4]); fr = cv2.imread(f)
    fr = cv2.inpaint(fr, spark, 12, cv2.INPAINT_NS)
    fr = (fr.astype(np.float32)*(1-ssoft) + cv2.GaussianBlur(fr, (0, 0), 5).astype(np.float32)*ssoft)
    M = Hs[n] @ T
    wl = cv2.warpPerspective(layer, M, (1920, 1080), flags=cv2.INTER_LINEAR)
    a = wl[..., 3]/255
    # lighting match: compare frame vs ref in the stripe area (dark/red pixels only, not occluded)
    pm = cv2.imread(f"{S}/wide/mask/{n:04d}.png", 0).astype(np.float32)/255
    occ = cv2.GaussianBlur(cv2.dilate((pm > 0.35).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32), (0, 0), 1.5)
    wref = cv2.warpPerspective(ref, Hs[n], (1920, 1080))
    sel = (a > 0.9) & (occ < 0.05) & (wref.max(2) < 150) & (fr.max(2) < 150)
    gain = np.ones(3)
    if sel.sum() > 300:
        lf, lr = fr[sel].mean(1), wref[sel].mean(1)            # brightness-only match, no colour shift
        gain = np.full(3, np.clip(np.median(lf) / (np.median(lr) + 1e-3), 0.8, 1.25))
    a = a * (1 - occ)
    out = fr*(1 - a[..., None]) + (wl[..., :3]*gain)*a[..., None]
    cv2.imwrite(f"{S}/wide/out/{n:04d}.png", np.clip(out, 0, 255).astype(np.uint8))
print("done")
