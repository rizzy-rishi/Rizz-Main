"""Final assembly: upscale + push-in + cinematic grade + bloom/halation + frosted end card
+ grain, then 60 fps title overlays and the score. Builds 4:5 and 9:16 masters."""
import json
import pathlib
import subprocess
import sys

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

R = pathlib.Path(__file__).parent.resolve()
W = R / "work"
FF = imageio_ffmpeg.get_ffmpeg_exe()
TL = json.loads((W / "timeline.json").read_text())
T = TL["total"]
E = 24.883333 - 4.0  # match title render timing
FPS = 60
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"


def titles(h):
    d = W / f"titles{h}"
    d.mkdir(exist_ok=True)
    n = int(T * FPS)
    if len(list(d.glob("*.png"))) >= n:
        return d
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME)
        pg = b.new_page(viewport={"width": 1080, "height": h})
        pg.goto((R / "titles.html").as_uri() + f"?h={h}&total={T}")
        pg.wait_for_function("window.READY===true")
        pg.evaluate("document.fonts.ready.then(()=>true)")
        pg.wait_for_timeout(300)
        for i in range(n):
            pg.evaluate(f"render({i / FPS})")
            pg.screenshot(path=str(d / f"{i:05d}.png"), omit_background=True)
        b.close()
    return d


GRADE = (
    "scale=1080:1936:flags=lanczos,crop=1080:1920,"
    f"scale=w='trunc(1080*(1+0.055*t/{T})/2)*2':h=-2:eval=frame:flags=bicubic,crop=1080:1920,"
    "unsharp=5:5:0.55:3:3:0,eq=saturation=1.1:contrast=1.02,format=gbrp,"
    "curves=master='0/0.025 0.22/0.19 0.5/0.5 0.78/0.81 1/0.975':"
    "r='0/0 0.5/0.525 1/1':g='0/0.01 0.5/0.5 1/0.985':b='0/0.055 0.5/0.48 1/0.92',"
    "colorbalance=rs=-0.03:gs=0.01:bs=0.04:rh=0.05:gh=0.02:bh=-0.04,"
    "split[a][b];[b]gblur=sigma=32,colorchannelmixer=rr=1:gg=0.88:bb=0.7[g];"
    "[a][g]blend=all_mode=screen:all_opacity=0.2,"
    "split[c][d];[d]gblur=sigma=48,curves=all='0/0.06 1/1'[fb];"
    f"[c][fb]blend=all_expr='A*(1-clip((T-{E})/1.1\\,0\\,1)*0.92)+B*clip((T-{E})/1.1\\,0\\,1)*0.92',"
    "vignette=angle=0.4,noise=alls=3:allf=t,format=yuv420p"
)


def build(h, out):
    tdir = titles(h)
    crop = "" if h == 1920 else ",crop=1080:1350:0:285"
    fc = f"[0:v]{GRADE}{crop}[v];[1:v]format=rgba[t];[v][t]overlay=0:0:shortest=1,format=yuv420p[o]"
    subprocess.run([FF, "-v", "error", "-y", "-i", str(W / "retimed.mp4"), "-framerate", str(FPS),
                    "-i", str(tdir / "%05d.png"), "-i", str(W / "score.wav"), "-filter_complex", fc,
                    "-map", "[o]", "-map", "2:a", "-r", str(FPS), "-c:v", "libx264", "-preset", "slow",
                    "-crf", "18", "-profile:v", "high", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
                    "-shortest", "-movflags", "+faststart", str(R / out)], check=True)
    print("built", out)


if __name__ == "__main__":
    which = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "titles":
        titles(1350 if which == "45" else 1920)
        sys.exit()
    if which == "45":
        build(1350, "Pangot-Valley-4x5-LinkedIn-X.mp4")
    else:
        build(1920, "Pangot-Valley-9x16-Reels.mp4")
