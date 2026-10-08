# Remove a static sparkle watermark from every frame of a video segment.
import cv2, numpy as np, subprocess, sys
src, out, ss, dur, cx, cy, r = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
W, H = 1920, 1080
mask = np.zeros((H, W), np.uint8)
t = np.linspace(0, 2*np.pi, 400)
x = cx + r*np.sign(np.cos(t))*np.abs(np.cos(t))**3
y = cy + r*np.sign(np.sin(t))*np.abs(np.sin(t))**3
cv2.fillPoly(mask, [np.stack([x, y], 1).astype(np.int32)], 255)
mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
soft = cv2.GaussianBlur(cv2.dilate(mask, np.ones((15, 15), np.uint8)).astype(np.float32)/255, (0, 0), 6)[..., None]
dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(ss), "-i", src, "-t", str(dur), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "24", "-i", "-",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
n = 0
while True:
    buf = dec.stdout.read(W*H*3)
    if len(buf) < W*H*3: break
    f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    f = cv2.inpaint(f, mask, 12, cv2.INPAINT_NS).astype(np.float32)
    f = f*(1-soft) + cv2.GaussianBlur(f, (0, 0), 5)*soft
    enc.stdin.write(f.clip(0, 255).astype(np.uint8).tobytes()); n += 1
enc.stdin.close(); enc.wait(); print("frames", n)
