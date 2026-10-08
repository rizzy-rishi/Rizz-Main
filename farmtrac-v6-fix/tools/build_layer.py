# Build the corrected side-decal layer (stripe patch + badges) in ref-frame space at 4x.
import cv2, numpy as np, sys
from PIL import Image, ImageDraw, ImageFont
S = sys.argv[1]; REF = int(sys.argv[2]); OUT = sys.argv[3]
SC = 4; X0, Y0, W, H = 680, 420, 340, 150           # canvas region in ref coords
FONT = "/home/user/Rizz-Main/logo-options/fonts/Unbounded_wght_800_.ttf"
ref = cv2.imread(f"{S}/wide/frames/{REF:04d}.png")
roi = ref[Y0:Y0+H, X0:X0+W].copy()

# 1) stripe patch: clean the garbled mark at the stripe tail, keep correct 6055 FARMTRAC
tail = np.zeros(roi.shape[:2], np.uint8)
cv2.rectangle(tail, (888-X0, 478-Y0), (930-X0, 503-Y0), 255, -1)
bright = (roi.max(2) > 120).astype(np.uint8)*255
fix = cv2.dilate(tail & bright, np.ones((3, 3), np.uint8))
roi_clean = cv2.inpaint(roi, fix, 4, cv2.INPAINT_TELEA)
stripe_poly = np.array([[703, 432], [936, 466], [936, 506], [703, 482]]) - [X0, Y0]
pm = np.zeros(roi.shape[:2], np.float32); cv2.fillPoly(pm, [stripe_poly.astype(np.int32)], 1)
pm = cv2.GaussianBlur(pm, (0, 0), 1.5)
big = cv2.resize(roi_clean, (W*SC, H*SC), interpolation=cv2.INTER_CUBIC).astype(np.float32)
alpha = cv2.resize(pm, (W*SC, H*SC), interpolation=cv2.INTER_LINEAR)
layer = np.dstack([big, alpha*255])                 # BGRA float, 0-255

# 2) badges rendered with PIL, sheared italic, mapped onto quads following the stripe slope
def text_img(parts, size, track=2):
    f = ImageFont.truetype(FONT, size)
    w = sum(f.getbbox(t)[2] + track for t, _ in parts) + 20; h = int(size*1.4)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x = 10
    for t, col in parts:
        d.text((x, int(size*0.1)), t, font=f, fill=col); x += f.getbbox(t)[2] + track
    im = im.crop(im.getbbox())
    sh = 0.22                                        # italic shear
    im = im.transform((int(im.width + sh*im.height), im.height), Image.AFFINE, (1, sh, -sh*im.height, 0, 1, 0), Image.BICUBIC)
    return np.array(im).astype(np.float32)
WHITE, RED = (204, 204, 210, 255), (190, 28, 38, 255)
def classic_pro():
    a = text_img([("CLASSIC", WHITE)], 120); b = text_img([("PRO", WHITE)], 104)
    h = a.shape[0] + b.shape[0] + 6; w = a.shape[1] + 10
    im = np.zeros((h, w, 4), np.float32); im[:a.shape[0], :a.shape[1]] = a
    im[a.shape[0]+6:, w-b.shape[1]-10:w-10] = b
    sw = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(sw)      # red swoosh + white rule
    y = a.shape[0] + 14
    d.polygon([(10, y+60), (w-b.shape[1]-40, y+6), (w-b.shape[1]-28, y+34), (24, y+84)], fill=RED)
    d.line([(4, y+98), (w-b.shape[1]-24, y+52)], fill=WHITE, width=9)
    s = np.array(sw).astype(np.float32); m = s[..., 3:]/255
    return im*(1-m) + s*m
def put(img, quad):
    """composite RGBA img (RGB order) onto layer at quad given in ref coords (tl,tr,br,bl)"""
    global layer
    h, w = img.shape[:2]
    q = (np.float32(quad) - [X0, Y0]) * SC
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), q.astype(np.float32))
    bgra = img[..., [2, 1, 0, 3]]
    wp = cv2.warpPerspective(bgra, M, (W*SC, H*SC), flags=cv2.INTER_AREA)
    a = wp[..., 3:]/255
    layer[..., :3] = layer[..., :3]*(1-a) + wp[..., :3]*a
    layer[..., 3:] = np.maximum(layer[..., 3:], wp[..., 3:])
def quad(x, y, w, h, slope):
    return [[x, y], [x+w, y+slope*w], [x+w, y+h+slope*w], [x, y+h]]
put(text_img([("4WD", WHITE)], 120), quad(737, 483, 27, 9, 0.07))
put(classic_pro(), quad(743, 514, 40, 18, 0.09))
put(text_img([("DARK ", WHITE), ("E", RED), ("DITION", WHITE)], 120), quad(878, 497, 50, 7.5, 0.18))
np.save(OUT, layer)
# preview on ref
lay = cv2.GaussianBlur(layer, (0, 0), 1.2)
small = cv2.resize(lay, (W, H), interpolation=cv2.INTER_AREA)
a = small[..., 3:]/255; prev = ref.copy().astype(np.float32)
prev[Y0:Y0+H, X0:X0+W] = prev[Y0:Y0+H, X0:X0+W]*(1-a) + small[..., :3]*a
cv2.imwrite(f"{S}/hood/layer_preview.png", cv2.resize(prev[Y0-10:Y0+H+30, X0-20:X0+W+20].astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
