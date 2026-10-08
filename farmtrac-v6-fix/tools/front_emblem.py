# Replace the blurry front "6055" emblem on the static hood plate with a crisp 60/55 badge built
# from the Farmtrac "6055" glyphs on the hood stripe (same brand numerals), finished in chrome.
import cv2, numpy as np, os
BX0, BY0 = 420, 380
LAYER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wide", "layer.npy")
def glyphs():
    L = np.load(LAYER)                                                  # 4x canvas, origin (680,420)
    c = L[(440-420)*4:(470-420)*4, (712-680)*4:(776-680)*4, :3]
    b, g, r = c[..., 0], c[..., 1], c[..., 2]
    a = np.clip((g - 110)/70, 0, 1) * np.clip((g - 0.55*r)/40, 0, 1)   # white text only, not red stripe
    n, lab, st, _ = cv2.connectedComponentsWithStats((a > 0.5).astype(np.uint8))
    keep = np.zeros_like(a)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_HEIGHT] > 0.6*a.shape[0]*0.6 and st[i, cv2.CC_STAT_AREA] > 300:
            keep[lab == i] = 1
    a = a*cv2.dilate(keep, np.ones((5, 5), np.uint8))
    ys, xs = np.where(a > 0.5); a = a[ys.min():ys.max()+1, xs.min():xs.max()+1]
    prof = (a > 0.5).sum(0); mid = a.shape[1]//2
    cut = mid - 25 + int(np.argmin(prof[mid-25:mid+25]))               # split between "0" and "5"
    return a[:, :cut], a[:, cut:]
def chrome(a):
    h, w = a.shape
    grad = np.linspace(222, 150, h)[:, None]*np.ones((1, w))           # light top, darker bottom
    inner = cv2.erode((a > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8))
    tone = np.where(inner > 0, grad, grad*0.8)                         # thin dark rim like a raised badge
    return np.dstack([tone, tone, tone*1.02, a*255]).astype(np.float32)
def fix(plate):
    p = plate.astype(np.uint8).copy()
    x0, y0, x1, y1 = 600, 552, 649, 582                                 # old emblem: remove bright pixels only
    sub = p[y0-BY0:y1-BY0, x0-BX0:x1-BX0]
    m = np.zeros(p.shape[:2], np.uint8)
    m[y0-BY0:y1-BY0, x0-BX0:x1-BX0] = (sub.max(2) > 45).astype(np.uint8)*255
    p = cv2.inpaint(p, cv2.dilate(m, np.ones((3, 3), np.uint8)), 3, cv2.INPAINT_TELEA).astype(np.float32)
    g60, g55 = glyphs()
    SC = 4; H60, H55 = 12*SC, 11*SC; WIDE = 1.45                       # real badge digits are wide and square
    bold = lambda g: cv2.dilate(g, np.ones((7, 7), np.uint8))           # chunkier strokes like the real badge
    g60, g55 = bold(g60), cv2.dilate(g55, np.ones((8, 8), np.uint8))
    s60 = cv2.resize(g60, (int(g60.shape[1]*H60/g60.shape[0]*WIDE), H60), interpolation=cv2.INTER_AREA)
    s55 = cv2.resize(g55, (int(g55.shape[1]*H55/g55.shape[0]*WIDE), H55), interpolation=cv2.INTER_AREA)
    W = int(s60.shape[1]*0.82) + s55.shape[1] + 2*SC; Hc = H60 + int(H55*0.68)
    canv = np.zeros((Hc, W, 4), np.float32)
    def put(rgba, x, y):
        al = rgba[..., 3:]/255; reg = canv[y:y+rgba.shape[0], x:x+rgba.shape[1]]
        reg[..., :3] = reg[..., :3]*(1-al) + rgba[..., :3]*al; reg[..., 3:] = np.maximum(reg[..., 3:], rgba[..., 3:])
    put(chrome(s55), int(s60.shape[1]*0.82), 0)                         # "55" raised, overlapping the 0
    put(chrome(s60), 0, Hc - H60)
    e = cv2.resize(canv, (W//SC, Hc//SC), interpolation=cv2.INTER_AREA)
    e = cv2.GaussianBlur(e, (0, 0), 0.4)
    h, w = e.shape[:2]; ox, oy = 601 - BX0, 554 - BY0
    a = e[..., 3:]/255
    p[oy:oy+h, ox:ox+w] = p[oy:oy+h, ox:ox+w]*(1-a) + e[..., :3]*a
    return p
