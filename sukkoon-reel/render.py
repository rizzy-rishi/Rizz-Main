"""Render index.html frame-by-frame to video (1080x1920, 30fps).

usage: python3 render.py stills 0.6 5 9 ...   -> preview PNGs in out/
       python3 render.py video                -> out/frames.mp4 (no audio)
"""
import pathlib
import subprocess
import sys

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).parent.resolve()
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
FPS, DUR = 30, 40.0
FF = imageio_ffmpeg.get_ffmpeg_exe()
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"


def open_page(p):
    browser = p.chromium.launch(executable_path=CHROME)
    page = browser.new_page(viewport={"width": 1080, "height": 1920})
    page.goto((ROOT / "index.html").as_uri() + "?render=1")
    page.wait_for_function("window.READY === true")
    page.evaluate("document.fonts.ready.then(() => true)")
    return browser, page


def main():
    mode = sys.argv[1]
    with sync_playwright() as p:
        browser, page = open_page(p)
        if mode == "stills":
            for t in map(float, sys.argv[2:]):
                page.evaluate(f"render({t})")
                page.screenshot(path=str(OUT / f"still_{t:05.2f}.png"))
        else:
            n = int(DUR * FPS)
            ff = subprocess.Popen(
                [FF, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS),
                 "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                 "-pix_fmt", "yuv420p", str(OUT / "frames.mp4")],
                stdin=subprocess.PIPE,
            )
            for i in range(n):
                page.evaluate(f"render({i / FPS})")
                ff.stdin.write(page.screenshot(type="jpeg", quality=93))
                if i % 150 == 0:
                    print("frame", i, flush=True)
            ff.stdin.close()
            ff.wait()
        browser.close()


main()
