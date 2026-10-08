# Person matte per frame with U2Net human-seg, refined with a guided filter on the frame.
import onnxruntime as ort, numpy as np, cv2, sys, glob, os
model, fdir, odir = sys.argv[1:4]
sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
def guided(I, p, r=8, eps=1e-3):
    box = lambda x: cv2.boxFilter(x, -1, (2*r+1, 2*r+1))
    mI, mp_ = box(I), box(p); a = (box(I*p) - mI*mp_) / (box(I*I) - mI*mI + eps); b = mp_ - a*mI
    return box(a)*I + box(b)
for f in sorted(glob.glob(fdir + "/*.png")):
    im = cv2.imread(f); h, w = im.shape[:2]
    x = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (320, 320)).astype(np.float32)/255
    x = ((x - mean)/std).transpose(2, 0, 1)[None].astype(np.float32)
    d = sess.run(None, {inp: x})[0][0, 0]
    d = (d - d.min())/(d.max() - d.min() + 1e-8)
    m = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    m = np.clip(guided(g, m), 0, 1)
    cv2.imwrite(os.path.join(odir, os.path.basename(f)), (m*255).astype(np.uint8))
