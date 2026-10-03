"""Speed-ramp the 240 fps optical-flow clip into a 60 fps timeline.

Speed (relative to real time) is set per source section and blended with
smoothstep ramps, so slow-downs and speed-ups glide instead of jumping.
"""
import json
import pathlib
import subprocess

import imageio_ffmpeg
import numpy as np

W = pathlib.Path(__file__).parent / "work"
FF = imageio_ffmpeg.get_ffmpeg_exe()
SRC_FPS, OUT_FPS = 240, 60

# (source start, speed): sky pan / golden sun hero / quick pan across the fog / final roofs
SECTIONS = [(0.0, 0.70), (3.6, 0.48), (8.8, 1.05), (11.0, 0.60)]
RAMP = 0.6  # seconds of source over which speed changes


def speed(s):
    v = SECTIONS[0][1]
    for (b, sp) in SECTIONS[1:]:
        p = np.clip((s - (b - RAMP / 2)) / RAMP, 0, 1)
        p = p * p * (3 - 2 * p)
        v = v + (sp - v) * p
    return v


def main():
    reader = imageio_ffmpeg.read_frames(str(W / "mi240.mp4"))
    meta = reader.__next__()
    w, h = meta["size"]
    n_src = int(meta["duration"] * SRC_FPS) - 2
    s_end = n_src / SRC_FPS
    ss = np.linspace(0, s_end, 20000)
    t_out = np.concatenate([[0], np.cumsum(np.diff(ss) / speed(ss[:-1]))])
    total = t_out[-1]
    n_out = int(total * OUT_FPS)
    src_idx = np.round(np.interp(np.arange(n_out) / OUT_FPS, t_out, ss) * SRC_FPS).astype(int)
    src_idx = np.clip(src_idx, 0, n_src - 1)

    wr = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                           "-r", str(OUT_FPS), "-i", "-", "-c:v", "libx264", "-crf", "10", "-preset", "fast",
                           "-pix_fmt", "yuv420p", str(W / "retimed.mp4")], stdin=subprocess.PIPE)
    want = iter(src_idx)
    nxt = next(want)
    for i, fr in enumerate(reader):
        while nxt == i:
            wr.stdin.write(fr)
            nxt = next(want, None)
            if nxt is None:
                break
        if nxt is None:
            break
    wr.stdin.close()
    wr.wait()
    # where each source section lands on the output timeline (for titles + music)
    marks = {f"src_{b}": float(np.interp(b, ss, t_out)) for b, _ in SECTIONS}
    marks["total"] = n_out / OUT_FPS
    (W / "timeline.json").write_text(json.dumps(marks, indent=1))
    print(marks)


main()
