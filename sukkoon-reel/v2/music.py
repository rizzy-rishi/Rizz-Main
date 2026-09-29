"""Kinetic 120 BPM soundtrack for the v2 reel (38 s, 44.1 kHz stereo).

Synthesised from scratch: tabla (dha/na/tin/ge), santoor plucks, bansuri lead,
pads, 808 sub, four-on-the-floor kick with sidechain pumping, and SFX hits placed
on the exact cut / text-slam times used in index.html.
"""
import pathlib
import wave

import numpy as np

SR = 44100
DUR = 38.0
N = int(SR * (DUR + 0.5))
rng = np.random.default_rng(11)
OUT = pathlib.Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
B = 0.5            # beat @120 BPM
S16 = B / 4
BAR = 4 * B


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tt(d):
    return np.arange(int(d * SR)) / SR


def buf():
    return np.zeros((2, N))


def place(bus, sig, t, pan=0.0, gain=1.0):
    i = int(round(t * SR))
    if i >= N or i < 0:
        return
    if sig.ndim == 1:
        sig = np.vstack([sig * np.cos((pan + 1) * np.pi / 4), sig * np.sin((pan + 1) * np.pi / 4)])
    n = min(sig.shape[1], N - i)
    bus[:, i:i + n] += gain * sig[:, :n]


def env(n, a=0.005, r=0.1):
    e = np.ones(n)
    ia, ir = min(int(a * SR), n), min(int(r * SR), n)
    if ia:
        e[:ia] = np.linspace(0, 1, ia)
    if ir:
        e[-ir:] *= np.linspace(1, 0, ir)
    return e


def filt(x, lo=None, hi=None, order=4):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    g = np.ones_like(f)
    if hi:
        g *= 1 / (1 + (f / hi) ** order)
    if lo:
        g *= 1 / (1 + (lo / np.maximum(f, 1e-3)) ** order)
    return np.fft.irfft(X * g, len(x))


def noise(d):
    return rng.standard_normal(int(d * SR))


def sweep(f0, f1, d, curve=1.0):
    t = tt(d)
    f = f0 + (f1 - f0) * (t / d) ** curve
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


# ------------------------------------------------------------------ drums
def kick(g=1.0):
    t = tt(0.45)
    body = np.sin(2 * np.pi * np.cumsum(48 + 110 * np.exp(-t * 32)) / SR) * np.exp(-t * 7.5)
    click = filt(noise(0.45), lo=2000) * np.exp(-t * 400) * 0.35
    return np.tanh((body + click) * 1.6) * 0.8 * g


def clap(g=1.0):
    t = tt(0.3)
    n = filt(noise(0.3), lo=1000, hi=6500)
    bursts = sum(np.exp(-np.maximum(t - k * 0.011, 0) * 180) * (t >= k * 0.011) for k in range(3))
    return n * (bursts * 0.5 + np.exp(-t * 18)) * 0.28 * g


def hat(g=1.0, d=0.05):
    t = tt(d)
    return filt(noise(d), lo=7500) * np.exp(-t * (90 if d < .1 else 25)) * 0.13 * g


def tabla(kind, g=1.0):
    t = tt(0.5)
    if kind in ("na", "tin", "dha"):
        f = 470 if kind != "tin" else 520
        dec = 9 if kind != "tin" else 16
        s = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * dec * r ** .5) for r, a in ((1, 1), (2.0, .5), (3.0, .3), (4.2, .15)))
        s += filt(noise(0.5), lo=2500) * np.exp(-t * 200) * 0.5
        out = s * 0.18
    else:
        out = np.zeros_like(t)
    if kind in ("ge", "dha"):
        f = 72 + 55 * (1 - np.exp(-t * 9))
        out += np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 5) * 0.42
    if kind == "ka":
        out = filt(noise(0.5), lo=300, hi=2500) * np.exp(-t * 60) * 0.25
    return out * g


def tom(g=1.0):
    t = tt(0.6)
    return (np.sin(2 * np.pi * np.cumsum(70 + 90 * np.exp(-t * 18)) / SR) * np.exp(-t * 6)
            + filt(noise(0.6), hi=2500) * np.exp(-t * 40) * 0.4) * 0.7 * g


def impact(g=1.0, d=2.8):
    t = tt(d)
    boom = np.sin(2 * np.pi * np.cumsum(34 + 70 * np.exp(-t * 5)) / SR) * np.exp(-t * 1.3)
    crack = filt(noise(d), lo=200, hi=5000) * np.exp(-t * 6) * 0.6
    return np.tanh((boom + crack) * 1.4) * 0.8 * g


def whoosh(d=0.35, g=1.0, up=True):
    n = noise(d)
    seg = 10
    L = len(n) // seg
    out = np.zeros_like(n)
    for k in range(seg):
        c = 400 + 5000 * ((k + .5) / seg if up else 1 - (k + .5) / seg) ** 1.5
        out[k * L:(k + 1) * L] = filt(n[k * L:(k + 1) * L], lo=c * .5, hi=c * 1.8)
    return out * np.sin(np.pi * np.linspace(0, 1, len(n))) ** 1.5 * 0.55 * g


def riser(d, g=1.0):
    t = tt(d)
    s = sweep(180, 2200, d, 2) * 0.22 + sweep(185, 2230, d, 2) * 0.22
    n = filt(noise(d), lo=1500) * 0.35
    return (s + n) * (t / d) ** 2.2 * g


def rev_cymbal(d=0.5, g=1.0):
    t = tt(d)
    return filt(noise(d), lo=5000) * (t / d) ** 3 * 0.45 * g


def tick(g=1.0, f=2600):
    t = tt(0.04)
    return np.sin(2 * np.pi * f * t) * np.exp(-t * 170) * 0.22 * g


def stamp(g=1.0):
    t = tt(0.2)
    return (np.sin(2 * np.pi * np.cumsum(75 + 150 * np.exp(-t * 60)) / SR) * np.exp(-t * 22)
            + filt(noise(0.2), lo=1200, hi=6000) * np.exp(-t * 90) * 0.7) * 0.5 * g


def blip(f=880, g=1.0):
    t = tt(0.22)
    return np.sin(2 * np.pi * np.cumsum(f * (1 + .5 * (1 - np.exp(-t * 50)))) / SR) * np.exp(-t * 20) * 0.22 * g


def zip_(g=1.0):
    return sweep(600, 2600, 0.28, 1.3) * np.exp(-tt(0.28) * 6) * 0.12 * g


def bell(m, d=3.0, g=1.0):
    t = tt(d)
    fc = mtof(m)
    return np.sin(2 * np.pi * fc * t + 2.2 * np.exp(-t * 2.5) * np.sin(2 * np.pi * fc * 3.5 * t)) * np.exp(-t * 1.5) * 0.2 * g


def heartbeat(g=1.0):
    t = tt(0.5)
    one = np.sin(2 * np.pi * 55 * t) * np.exp(-t * 18)
    two = np.zeros_like(t)
    k = int(0.16 * SR)
    two[k:] = one[:-k] * 0.7
    return (one + two) * 0.6 * g


# ------------------------------------------------------------------ tonal
def pluck(m, d=1.2, g=1.0):
    f = mtof(m)
    p = max(2, int(round(SR / f)))
    n = int(d * SR)
    y = np.zeros(n + p + 1)
    y[:p] = filt(rng.uniform(-1, 1, p * 4), hi=6000)[:p]
    for i in range(p, n + p, p):
        seg = y[i - p:i + 1]
        blk = 0.5 * (seg[:-1] + seg[1:]) * 0.997
        y[i:i + p] = blk[:len(y[i:i + p])]
    return y[:n] * env(n, 0.001, 0.2) * 0.33 * g


def pad(m, d, g=1.0):
    t = tt(d)
    out = np.zeros_like(t)
    for det in (-0.08, 0, 0.08):
        f = mtof(m + det)
        for k in range(1, 8):
            out += np.sin(2 * np.pi * f * k * t + rng.random() * 6.28) / k * np.exp(-k / 3.5)
    return out * env(len(t), 0.5, 0.6) * 0.06 * g


def flute(m, d, g=1.0):
    t = tt(d)
    f = mtof(m) * (1 + 0.007 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - .15) * 4, 0, 1))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) + .2 * np.sin(2 * ph) + .07 * np.sin(3 * ph)
    s += filt(noise(d), lo=1500, hi=6000) * 0.05
    return s * env(len(t), 0.06, min(0.25, d * .4)) * 0.12 * g


def sub(m, d, g=1.0):
    t = tt(d)
    f = mtof(m)
    s = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)
    return np.tanh(s * 1.3) * env(len(t), 0.004, 0.06) * 0.42 * g


# ------------------------------------------------------------------ arrangement
mus, drm, sfx = buf(), buf(), buf()
PROG = [[47, 50, 54, 59], [43, 47, 50, 55], [50, 54, 57, 62], [45, 49, 52, 57]]  # Bm G D A


def chord_at(t):
    return PROG[int(t // BAR) % 4]


def grid(a, b, step):
    x = np.ceil(a / step - 1e-9) * step
    out = []
    while x < b - 1e-6:
        out.append(round(float(x), 5))
        x += step
    return out


# sections where the groove plays
GROOVE = [(4, 12, 1.0), (14, 23, 1.0), (25, 28.5, 1.15), (28.5, 31, 1.0)]
KICKS = []
for a, b, g in GROOVE:
    for x in grid(a, b, B):
        KICKS.append(x)
        place(drm, kick(g), x)
for x in grid(31, 35, 2 * B):   # half-time outro
    KICKS.append(x)
    place(drm, kick(0.8), x)

for a, b, g in [(4, 12, 1), (14, 23, 1), (25, 28.5, 1.1)]:
    for x in grid(a + B, b, 2 * B):
        place(drm, clap(g), x, pan=-.05)
    for x in grid(a, b, B / 2):
        place(drm, hat(.9 if (x / (B / 2)) % 2 else .5), x, pan=.35)
for x in grid(25, 28.5, S16):
    place(drm, hat(.5), x, pan=.35)
for x in grid(15, 23, B):
    place(drm, hat(.6, 0.25), x + B / 2, pan=-.35)   # open hats on the off-beat

TABLA = {0: "dha", 2: "ge", 3: "na", 4: "tin", 6: "na", 7: "ka", 8: "dha", 10: "ge", 11: "na", 12: "tin", 13: "na", 14: "dha", 15: "ka"}
for a, b, g in [(2, 12, .9), (14, 23, 1), (25, 31, 1.1), (31, 35, .7)]:
    for x in grid(a, b, S16):
        k = int(round((x % BAR) / S16)) % 16
        if k in TABLA:
            place(drm, tabla(TABLA[k], g * (1.0 if k % 4 == 0 else .75)), x, pan=.15)

# sub / bass: 8ths following the chord root
for a, b in [(4, 12), (14, 23), (25, 31)]:
    for x in grid(a, b, B / 2):
        root = chord_at(x)[0] - 12
        place(mus, sub(root, B / 2 * .9, 1.0 if (x / (B / 2)) % 2 == 0 else .7), x)
for x in [14.0, 16.0, 31.0]:
    place(mus, sub(chord_at(x)[0] - 24, 1.6, 1.3), x)

# pads everywhere except the "but" stop
for x in grid(0, 38, BAR):
    if 22.9 <= x < 25:
        continue
    for m in chord_at(x):
        place(mus, pad(m, BAR + .6, 1.3 if 12 <= x < 14 or x >= 31 else 1), x, pan=rng.uniform(-.5, .5))

# santoor: 16th arpeggio in hook build + drops, 8ths elsewhere
for a, b, step, g in [(2, 4, S16, .6), (4, 12, B / 2, .9), (12, 14, B / 2, .6), (14, 23, S16, .7), (25, 31, S16, .7)]:
    for j, x in enumerate(grid(a, b, step)):
        ch = chord_at(x)
        m = [ch[0] + 24, ch[2] + 12, ch[1] + 24, ch[3] + 12][j % 4]
        place(mus, pluck(m, .9, g * (1 if j % 2 == 0 else .7)), x, pan=-.4 + .27 * (j % 4))

# bansuri hook (Bm / D pentatonic)
MEL = [(4.0, 78, .45), (4.5, 81, .45), (5.0, 83, .9), (6.0, 81, .45), (6.5, 78, .45), (7.0, 76, .9),
       (8.0, 78, .45), (8.5, 81, .45), (9.0, 86, .9), (10.0, 83, .45), (10.5, 81, .45), (11.0, 78, 1.0),
       (12.2, 74, .7), (12.9, 76, .5), (13.4, 78, .6),
       (18.0, 81, .45), (18.5, 83, .45), (19.0, 86, .9), (20.0, 83, .45), (20.5, 81, .45), (21.0, 78, 1.8),
       (31.5, 78, .5), (32.0, 81, .5), (32.5, 83, 1.0), (34.0, 81, .5), (34.5, 78, .5), (35.0, 74, 2.5)]
for x, m, d in MEL:
    place(mus, flute(m, d + .2), x, pan=.12)
for k, m in enumerate([86, 83, 81, 78, 74, 71]):
    place(mus, pluck(m, 2.0, .8), 33.0 + k * .25, pan=-.3 + .12 * k)

# ------------------------------------------------------------------ SFX on picture
place(sfx, impact(1.0), 0.0)
for x, g in [(0, 1), (.5, .8), (1.0, .8), (1.5, 1.1)]:
    place(sfx, tom(g), x)
place(sfx, riser(1.0, .8), 3.0)
for x in [3.85, 4.85, 5.85, 6.85, 7.85, 8.85]:
    place(sfx, whoosh(.3, 1.0), x, pan=.4)
for x in [10.0, 10.5, 11.0, 11.5]:
    place(sfx, tom(.8), x)
    place(sfx, clap(.7), x)
place(sfx, impact(.7), 12.0)
place(sfx, riser(1.0, .9), 13.0)
place(sfx, impact(1.2), 14.0)
for x in [14.25, 15.0, 15.25, 16.0, 16.25, 17.0, 17.25]:
    place(sfx, tom(.9 if x % 1 == 0 else .6), x)
place(sfx, whoosh(.3, 1.0, up=False), 17.85)
place(sfx, blip(660, .9), 18.5)
for i in range(4):
    place(sfx, zip_(1.0), 19 + i * .5, pan=[-.5, .5, .5, -.5][i])
    place(sfx, blip(880 + 110 * i, .8), 19.22 + i * .5, pan=[-.5, .5, .5, -.5][i])
# tape-stop into "but"
place(sfx, sweep(300, 40, .25) * np.exp(-tt(.25) * 8) * .3, 22.75)
place(sfx, tom(.6), 23.0)
for x in grid(23.0, 25.0, B / 2):
    place(sfx, tick(.9, 2600 if (x / (B / 2)) % 2 == 0 else 2000), x, pan=.2)
place(sfx, heartbeat(1.0), 23.0)
place(sfx, heartbeat(1.1), 24.0)
place(sfx, tom(.9), 23.5)
place(sfx, tom(1.0), 24.0)
place(sfx, rev_cymbal(.5, 1.0), 24.0)
place(sfx, impact(1.3), 24.5)
place(sfx, riser(.4, .6), 24.6)
place(sfx, impact(.6), 25.0)
n_sold = 20
for i in range(n_sold):
    place(sfx, stamp(1.0), 25.5 + 2.0 * np.sqrt(i / (n_sold - 1)), pan=rng.uniform(-.3, .3))
for x in grid(25, 27.75, S16):
    place(sfx, tick(.5), x, pan=.25)
place(sfx, impact(1.0), 27.75)
place(sfx, tom(1.0), 28.0)
for x in [28.5, 29.0, 29.5, 30.0]:
    place(sfx, impact(.55, 1.0), x)
    place(sfx, clap(1.0), x)
place(sfx, riser(.8, .9), 30.2)
place(sfx, impact(1.3), 31.0)
place(sfx, bell(83, 3.0, 1.0), 32.1, pan=.1)
place(sfx, bell(90, 2.5, .5), 32.35, pan=-.2)
place(sfx, blip(1175, .7), 32.5)

# ------------------------------------------------------------------ mix + master
def sidechain(x, depth=0.55, rel=0.14):
    g = np.ones(x.shape[1])
    t = np.arange(x.shape[1]) / SR
    for k in KICKS:
        i = int(k * SR)
        j = min(len(g), i + int(rel * 4 * SR))
        g[i:j] = np.minimum(g[i:j], 1 - depth * np.exp(-(t[i:j] - k) / rel))
    return x * g


def reverb(x, secs, wet):
    n = int(secs * SR)
    t = np.arange(n) / SR
    out = np.zeros_like(x)
    for ch in range(2):
        ir = filt(rng.standard_normal(n) * np.exp(-t * 6.9 / secs), hi=7000)
        ir /= np.sqrt(np.sum(ir ** 2))
        L = x.shape[1] + n
        out[ch] = np.fft.irfft(np.fft.rfft(x[ch], L) * np.fft.rfft(ir, L), L)[:x.shape[1]]
    return x * (1 - wet) + out * wet * 1.3


mus = reverb(sidechain(mus), 2.4, .3)
drm = reverb(drm, 0.9, .08)
sfx = reverb(sfx, 1.6, .18)
mix = mus + drm * 0.95 + sfx

# simple bus compressor (RMS, 3:1 above -18 dBFS)
mono = np.mean(np.abs(mix), axis=0)
w = int(0.02 * SR)
cs = np.cumsum(np.concatenate([[0], mono ** 2]))
rms = np.sqrt(np.maximum((cs[w:] - cs[:-w]) / w, 1e-12))
rms = np.concatenate([rms, np.full(len(mono) - len(rms), rms[-1])])
lvl = 20 * np.log10(rms / (np.max(rms) + 1e-12))
gr = np.where(lvl > -18, (lvl + 18) * (1 - 1 / 3), 0)
mix *= 10 ** (-gr / 20)

fi = int(.03 * SR)
mix[:, :fi] *= np.linspace(0, 1, fi)
a, b = int(36.3 * SR), int(DUR * SR)
mix[:, a:b] *= np.linspace(1, 0, b - a) ** 1.5
mix = mix[:, :b]
mix *= 10 ** (-14.5 / 20) / np.sqrt(np.mean(mix ** 2))
mix = np.tanh(mix * 1.15) / np.tanh(1.15)
mix *= 0.95 / np.max(np.abs(mix))

with wave.open(str(OUT / "music.wav"), "wb") as f:
    f.setnchannels(2)
    f.setsampwidth(2)
    f.setframerate(SR)
    f.writeframes((mix.T * 32767).astype(np.int16).tobytes())
print("ok", 20 * np.log10(np.sqrt(np.mean(mix ** 2))))
