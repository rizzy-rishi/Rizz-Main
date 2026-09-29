"""Cut, grade and scale the site footage/photos into JPEG frame sequences for the v2 reel.

Output: assets/<name>/0001.jpg ... (30 fps) and assets/photos/<name>.jpg
Sources are the raw uploads (phone footage + WhatsApp clips + photos).
"""
import pathlib
import subprocess
import sys

import imageio_ffmpeg
from PIL import Image, ImageEnhance, ImageFilter

FF = imageio_ffmpeg.get_ffmpeg_exe()
UP = pathlib.Path("/root/.claude/uploads/8be0f7fd-de2b-58d9-9332-490a70bfa8bf")
IMG = pathlib.Path("/tmp/claude-0/-home-user-Rizz-Main/8be0f7fd-de2b-58d9-9332-490a70bfa8bf/images")
OUT = pathlib.Path(__file__).parent / "assets"

V_4K = UP / "53107204-video_20260928_112836.mp4"               # misty railing -> roofs (4K)
V_SUN = UP / "276e9dcc-WhatsApp_Video_2026-09-29_at_6.32.13_PM.mp4"  # clear sky, deodars, valley
V_MIST = UP / "cf3e3025-WhatsApp_Video_2026-09-29_at_6.32.40_PM.mp4"  # mist + wooden cottage
V_FOG = UP / "c3fed2d8-WhatsApp_Video_2026-09-29_at_6.33.26_PM.mp4"   # fog over roofs, deodars

# Colour grade: lift contrast/saturation, cool shadows, warm highlights (brand teal + gold)
MIST_GRADE = "eq=contrast=1.16:brightness=-0.015:saturation=1.35:gamma=0.96,colorbalance=bs=0.05:bm=0.015:rh=0.045:gh=0.01:bh=-0.02"
SUN_GRADE = "eq=contrast=1.08:saturation=1.18:gamma=0.98,colorbalance=rh=0.03:bh=-0.02"
LOWRES_FIX = "hqdn3d=1.5:1.5:3:3"
SHARP = "unsharp=5:5:0.7:5:5:0.0"

SEGMENTS = [
    # name, src, start, dur, filter chain
    ("hero", V_4K, 0.0, 9.4, f"crop=1215:2160:{int(sys.argv[1]) if len(sys.argv) > 1 else 1312}:0,scale=1080:1920:flags=lanczos,{MIST_GRADE}"),
    ("cot", V_MIST, 14.0, 6.0, f"{LOWRES_FIX},scale=1080:608:flags=lanczos,{MIST_GRADE},{SHARP}"),
    ("mistroof", V_MIST, 0.0, 4.0, f"{LOWRES_FIX},scale=1080:608:flags=lanczos,{MIST_GRADE},{SHARP}"),
    ("sunny1", V_SUN, 0.0, 6.0, f"{LOWRES_FIX},scale=1080:608:flags=lanczos,{SUN_GRADE},{SHARP}"),
    ("sunny2", V_SUN, 20.0, 8.0, f"{LOWRES_FIX},scale=1080:608:flags=lanczos,{SUN_GRADE},{SHARP}"),
    ("fog1", V_FOG, 0.0, 4.0, f"{LOWRES_FIX},scale=1080:608:flags=lanczos,{MIST_GRADE},{SHARP}"),
    ("fog2", V_FOG, 20.5, 4.5, f"{LOWRES_FIX},scale=1080:608:flags=lanczos,{MIST_GRADE},{SHARP}"),
]

# 4:5 crops (x0, y0, w, h) chosen to avoid laundry / clutter in the frames
PHOTOS = {
    "deck": ("14.jpg", (1210, 0, 722, 903)),
    "roofs": ("12.jpg", (740, 0, 722, 903)),
    "deodar": ("11.jpg", (930, 0, 722, 903)),
    "balcony": ("10.jpg", (1060, 0, 722, 903)),
}


def run(cmd):
    subprocess.run(cmd, check=True)


def segments(only=None):
    for name, src, ss, dur, vf in SEGMENTS:
        if only and name not in only:
            continue
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.jpg"):
            f.unlink()
        run([FF, "-v", "error", "-ss", str(ss), "-t", str(dur), "-i", str(src), "-vf", f"{vf},fps=30",
             "-q:v", "2", str(d / "%04d.jpg")])
        print(name, len(list(d.glob("*.jpg"))), "frames")


def photos():
    d = OUT / "photos"
    d.mkdir(parents=True, exist_ok=True)
    for name, (fn, (x, y, w, h)) in PHOTOS.items():
        im = Image.open(IMG / fn).convert("RGB").crop((x, y, x + w, y + h))
        im = im.resize((1080, 1350), Image.LANCZOS)
        im = ImageEnhance.Contrast(im).enhance(1.14)
        im = ImageEnhance.Color(im).enhance(1.3)
        im = im.filter(ImageFilter.UnsharpMask(radius=2.2, percent=70, threshold=2))
        im.save(d / f"{name}.jpg", quality=93)
        print("photo", name)


def logo():
    """Logo with the white background knocked out (for use on footage)."""
    im = Image.open(IMG / "1.png").convert("RGBA")
    px = im.load()
    for yy in range(im.height):
        for xx in range(im.width):
            r, g, b, a = px[xx, yy]
            m = min(r, g, b)
            if m > 235:
                px[xx, yy] = (r, g, b, 0)
            elif m > 200:
                px[xx, yy] = (r, g, b, int(a * (235 - m) / 35))
    im = im.crop(im.getbbox())
    im.save(OUT / "logo.png")
    print("logo", im.size)


if __name__ == "__main__":
    segments()
    photos()
    logo()
