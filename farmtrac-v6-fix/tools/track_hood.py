# Track the hood side panel: homography ref->frame for every frame (SIFT, person-masked, smoothed).
import cv2, numpy as np, glob, os, sys, json
S = sys.argv[1]; REF = int(sys.argv[2])
frames = sorted(glob.glob(f"{S}/wide/frames/*.png")); idx = [int(os.path.basename(f)[:4]) for f in frames]
poly = np.array([[690, 415], [1010, 440], [1010, 610], [690, 610]], np.float32)   # side panel in ref
sift = cv2.SIFT_create(4000)
def load(n):
    g = cv2.cvtColor(cv2.imread(f"{S}/wide/frames/{n:04d}.png"), cv2.COLOR_BGR2GRAY)
    pm = cv2.imread(f"{S}/wide/mask/{n:04d}.png", 0)
    return g, (pm < 100).astype(np.uint8)
def region(polygon, notperson, grow=60):
    m = np.zeros_like(notperson); cv2.fillPoly(m, [polygon.astype(np.int32)], 1)
    m = cv2.dilate(m, np.ones((grow, grow), np.uint8)); return (m & notperson)*255
gr, npr = load(REF)
kr, dr = sift.detectAndCompute(gr, region(poly, npr, 1))
bf = cv2.BFMatcher()
H = {REF: np.eye(3)}
def match(n, Hprev):
    g, npm = load(n)
    k, d = sift.detectAndCompute(g, region(cv2.perspectiveTransform(poly[None], Hprev)[0], npm))
    if d is None or len(k) < 10: return None, 0
    ms = [m for m, m2 in bf.knnMatch(dr, d, k=2) if m.distance < 0.75*m2.distance]
    if len(ms) < 10: return None, 0
    src = np.float32([kr[m.queryIdx].pt for m in ms]); dst = np.float32([k[m.trainIdx].pt for m in ms])
    h, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return h, int(inl.sum()) if inl is not None else 0
log = {}
for direction in (range(REF+1, idx[-1]+1), range(REF-1, idx[0]-1, -1)):
    prev = REF
    for n in direction:
        h, ni = match(n, H[prev]); log[n] = ni
        H[n] = h if (h is not None and ni >= 12) else H[prev]
        prev = n
# temporal smoothing on projected corners
ns = sorted(H); C = np.array([cv2.perspectiveTransform(poly[None], H[n])[0] for n in ns])
good = np.array([n == REF or log.get(n, 0) >= 30 for n in ns])
nsa = np.array(ns)
for j in range(4):                     # interpolate unreliable frames from reliable neighbours
    for k in range(2):
        C[~good, j, k] = np.interp(nsa[~good], nsa[good], C[good, j, k])
json.dump({int(n): int(log.get(n, 999)) for n in ns}, open(f"{S}/wide/inliers.json", "w"))
from scipy.ndimage import gaussian_filter1d
Cs = gaussian_filter1d(C, sigma=1.5, axis=0, mode="nearest")
out = {int(n): cv2.getPerspectiveTransform(poly, Cs[i].astype(np.float32)).tolist() for i, n in enumerate(ns)}
json.dump(out, open(f"{S}/wide/H.json", "w"))
print("min inliers", min(log.values()), "frames <30:", [n for n, v in log.items() if v < 30][:40])
