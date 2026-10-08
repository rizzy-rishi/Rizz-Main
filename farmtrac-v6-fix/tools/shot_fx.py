# Shot-1 frame pass: sparkle removal, drifting fog in the blue haze, subtle hair flutter.
import cv2, numpy as np, subprocess, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--ss", type=float); ap.add_argument("--dur", type=float)
ap.add_argument("--fog", type=float, default=14.0)   # fog modulation strength (0-255 levels)
ap.add_argument("--hair", type=float, default=2.0)   # max hair displacement in px
ap.add_argument("--debug", default="")
a = ap.parse_args()
W, H, FPS = 1920, 1080, 24
rng = np.random.default_rng(7)

# sparkle mask (fixed position)
cx, cy, r = 1740, 900, 36
spark = np.zeros((H, W), np.uint8)
t = np.linspace(0, 2*np.pi, 400)
pts = np.stack([cx + r*np.sign(np.cos(t))*np.abs(np.cos(t))**3, cy + r*np.sign(np.sin(t))*np.abs(np.sin(t))**3], 1)
cv2.fillPoly(spark, [pts.astype(np.int32)], 255)
spark = cv2.dilate(spark, np.ones((7, 7), np.uint8))
spark_soft = cv2.GaussianBlur(cv2.dilate(spark, np.ones((15, 15), np.uint8)).astype(np.float32)/255, (0, 0), 6)[..., None]

# fog noise fields, larger than the frame so they can drift
PAD = 600
def field(cell):
    g = rng.random(((H+PAD)//cell + 3, (W+PAD)//cell + 3)).astype(np.float32)
    f = cv2.resize(g, ((W+PAD)//cell*cell + 3*cell, (H+PAD)//cell*cell + 3*cell), interpolation=cv2.INTER_CUBIC)
    f = cv2.GaussianBlur(f, (0, 0), cell/3)
    return (f - f.mean()) / (f.std() + 1e-6)
coarseA, coarseB, fine = field(160), field(160), field(70)
fog_col = np.array([1.0, 0.86, 0.72], np.float32)  # BGR, cool haze

def fog_layer(i):
    s = i / FPS
    ox, oy = int(PAD - 22*s) , int(PAD/2 - 5*s)      # slow drift right->left and slightly up
    fx = int(PAD - 38*s)
    w = 0.5 + 0.5*np.sin(2*np.pi*s/6.0)
    n = (1-w)*coarseA[oy:oy+H, ox:ox+W] + w*coarseB[oy:oy+H, ox:ox+W]
    n = 0.7*n + 0.3*fine[oy:oy+H, fx:fx+W]
    return np.clip(n, -2.5, 2.5) / 2.5

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
face_box = None

def hair_warp(f, i):
    global face_box
    small = cv2.resize(f, (W//4, H//4))
    b, g, rr = [small[..., k].astype(np.int16) for k in range(3)]
    v = small.max(2).astype(np.int16)
    skin = ((rr > b + 25) & (rr > 70) & (v > 70)).astype(np.uint8)
    skin[620//4:, :] = 0                                   # face only, never the hand on the bonnet
    n, lab, st, _ = cv2.connectedComponentsWithStats(skin)
    if n > 1:
        k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
        x0, y0, bw, bh = [st[k, j]*4 for j in (0, 1, 2, 3)]
        box = np.array([x0, y0, x0+bw, y0+bh], np.float32)
        face_box = box if face_box is None else 0.6*face_box + 0.4*box
    if face_box is None: return f
    x0, y0, x1, y1 = face_box; fw, fh = x1-x0, y1-y0
    zx0, zx1 = int(max(0, x0 - 0.35*fw)), int(min(W, x1 + 0.35*fw))
    zy0, zy1 = int(max(0, y0 - 0.9*fh)), int(min(H, y0 + 0.12*fh))
    zone = np.zeros((H, W), np.float32); zone[zy0:zy1, zx0:zx1] = 1
    dark = (f.max(2) < 70).astype(np.float32) * zone
    m = cv2.GaussianBlur(cv2.dilate(dark, np.ones((9, 9), np.uint8)), (0, 0), 4)
    ramp = np.clip((y0 + 0.12*fh - yy) / (0.9*fh + 1e-6), 0, 1) ** 0.7   # stronger toward the crown
    wgt = (m * ramp)
    s = i / FPS
    n1 = 0.6*np.sin(0.035*xx + 0.021*yy + 2*np.pi*0.8*s) + 0.4*np.sin(0.061*xx - 0.033*yy + 2*np.pi*1.3*s + 1.0)
    n2 = 0.6*np.sin(0.029*xx - 0.017*yy + 2*np.pi*0.7*s + 2.0) + 0.4*np.sin(0.05*xx + 0.04*yy + 2*np.pi*1.1*s)
    mapx = xx + a.hair*wgt*n1
    mapy = yy + 0.5*a.hair*wgt*n2
    return cv2.remap(f, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(a.ss), "-i", a.src, "-t", str(a.dur), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
i = 0
while True:
    buf = dec.stdout.read(W*H*3)
    if len(buf) < W*H*3: break
    f = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
    f = cv2.inpaint(f, spark, 12, cv2.INPAINT_NS)
    f = (f.astype(np.float32)*(1-spark_soft) + cv2.GaussianBlur(f, (0, 0), 5).astype(np.float32)*spark_soft).clip(0, 255).astype(np.uint8)
    if a.hair > 0: f = hair_warp(f, i)
    ff = f.astype(np.float32)
    if a.fog > 0:
        b, g, rr = ff[..., 0], ff[..., 1], ff[..., 2]
        key = np.clip((b - rr - 8)/30, 0, 1) * np.clip((ff.max(2) - 25)/40, 0, 1)
        key = cv2.GaussianBlur(key, (0, 0), 20)[..., None]
        ff = ff + a.fog * fog_layer(i)[..., None] * key * fog_col
    out = ff.clip(0, 255).astype(np.uint8)
    if a.debug and i % 6 == 0: cv2.imwrite(f"{a.debug}/f{i:03d}.png", out)
    enc.stdin.write(out.tobytes()); i += 1
enc.stdin.close(); enc.wait(); print("frames", i, "face_box", face_box)
