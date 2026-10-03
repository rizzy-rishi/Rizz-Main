"""Cinematic ambient score for the Pangot valley reel, cut to the title timings.

Felt-piano motif, warm string pads, bansuri, santoor sparkle, reverse swells into
each title, soft booms on the reveals, a riser and chimes for the logo end card.
"""
import json
import pathlib
import wave

import numpy as np

R = pathlib.Path(__file__).parent
TL = json.loads((R / "work" / "timeline.json").read_text())
DUR = TL["total"]
E = DUR - 4.0
MARKS = [0.45, 6.1, 12.5, 18.1, E]
SR = 48000
N = int(SR * (DUR + 4))
rng = np.random.default_rng(21)


def mtof(m):
    return 440 * 2 ** ((m - 69) / 12)


def tt(d):
    return np.arange(int(d * SR)) / SR


def filt(x, lo=None, hi=None, o=4):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    g = np.ones_like(f)
    if hi:
        g /= 1 + (f / hi) ** o
    if lo:
        g /= 1 + (lo / np.maximum(f, 1e-3)) ** o
    return np.fft.irfft(X * g, len(x))


def env(n, a, r, curve=2):
    e = np.ones(n)
    ia, ir = min(int(a * SR), n), min(int(r * SR), n)
    if ia:
        e[:ia] = np.linspace(0, 1, ia) ** curve
    if ir:
        e[n - ir:] *= np.linspace(1, 0, ir) ** curve
    return e


mus, fx = np.zeros((2, N)), np.zeros((2, N))


def place(bus, sig, t, pan=0.0, g=1.0):
    i = int(t * SR)
    if i >= N:
        return
    if sig.ndim == 1:
        sig = np.vstack([sig * np.cos((pan + 1) * np.pi / 4), sig * np.sin((pan + 1) * np.pi / 4)])
    n = min(sig.shape[1], N - i)
    bus[:, i:i + n] += g * sig[:, :n]


def strings(m, d):
    t = tt(d)
    out = np.zeros_like(t)
    for det in (-.09, -.03, .03, .09):
        f = mtof(m + det) * (1 + .002 * np.sin(2 * np.pi * (4.5 + det * 10) * t))
        ph = 2 * np.pi * np.cumsum(f) / SR + rng.random() * 6
        out += sum(np.sin(k * ph) / k * np.exp(-k / 3.2) for k in range(1, 9))
    out = filt(out, 80, 3200)
    return out * env(len(t), 1.6, 2.0) * .016


def piano(m, d=4.0, v=1.0):
    t = tt(d)
    f = mtof(m)
    s = sum(a * np.sin(2 * np.pi * f * r * t + .3 * np.sin(2 * np.pi * f * t) * np.exp(-t * 6))
            * np.exp(-t * (1.1 + .9 * r)) for r, a in ((1, 1), (2.01, .42), (3.02, .16), (4.05, .07)))
    ham = filt(rng.standard_normal(len(t)), 800, 5000) * np.exp(-t * 90) * .05
    return (s + ham) * env(len(t), .003, .5) * .085 * v


def santoor(m, d=2.5):
    t = tt(d)
    f = mtof(m)
    s = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * (2.2 + r)) for r, a in ((1, 1), (2.0, .5), (3.0, .25), (5.2, .1)))
    return s * env(len(t), .002, .3) * .05


def flute(m, d):
    t = tt(d)
    f = mtof(m) * (1 + .007 * np.sin(2 * np.pi * 5.2 * t) * np.clip((t - .3) * 3, 0, 1))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) + .14 * np.sin(2 * ph) + filt(rng.standard_normal(len(t)), 1800, 6000) * .06
    return s * env(len(t), .3, min(.8, d * .5)) * .06


def boom(g=1.0):
    t = tt(4.0)
    s = np.sin(2 * np.pi * np.cumsum(32 + 26 * np.exp(-t * 3)) / SR) * np.exp(-t * 1.1)
    air = filt(rng.standard_normal(len(t)), 300, 4000) * np.exp(-t * 3.5) * .25
    return np.tanh((s + air) * 1.2) * .35 * g


def reverse_swell(d=1.6, g=1.0):
    t = tt(d)
    n = filt(rng.standard_normal(len(t)), 500, 9000) * (t / d) ** 3
    tone = sum(np.sin(2 * np.pi * mtof(m) * t) for m in (62, 69, 74)) / 3 * (t / d) ** 4 * .4
    return (n * .22 + tone) * g


def riser(d, g=1.0):
    t = tt(d)
    s = np.sin(2 * np.pi * np.cumsum(300 + 1500 * (t / d) ** 2.2) / SR) * .12
    n = filt(rng.standard_normal(len(t)), 2000, 12000) * .25
    return (s + n) * (t / d) ** 2.5 * g


def chime(m, g=1.0):
    t = tt(5.0)
    f = mtof(m)
    s = np.sin(2 * np.pi * f * t + 1.6 * np.exp(-t * 1.8) * np.sin(2 * np.pi * f * 2.76 * t)) * np.exp(-t * .9)
    return s * env(len(t), .002, 1.0) * .07 * g


def chirp(g=1.0):
    t = tt(.1)
    f = rng.uniform(3200, 4500) + rng.uniform(600, 1300) * np.sin(np.pi * t / .1) * rng.choice([-1, 1])
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / .1) ** 2 * .018 * g


# harmony per section: Dmaj9 | Bm9 | Gmaj9 | Asus2->A | Dmaj9 (resolve)
CHORDS = [[50, 57, 61, 64, 66], [47, 54, 57, 61, 62], [43, 50, 54, 57, 62], [45, 52, 57, 59, 64], [38, 50, 57, 61, 64, 66]]
bounds = MARKS + [DUR + 2]
for i, ch in enumerate(CHORDS):
    a, b = bounds[i] - (0.45 if i else 0.45), bounds[i + 1] + 1.6
    for m in ch:
        place(mus, strings(m, b - a), a, pan=rng.uniform(-.7, .7))

# felt-piano motif: gentle broken chords, every 0.86 s
for i, ch in enumerate(CHORDS[:4]):
    a, b = MARKS[i], MARKS[i + 1] - .2
    pat = [ch[0] + 24, ch[2] + 12, ch[3] + 12, ch[1] + 24, ch[4] + 12, ch[2] + 12]
    k, t = 0, a
    while t < b:
        place(mus, piano(pat[k % len(pat)], 3.6, .85 if k % 2 else 1.0), t, pan=-.3 + .12 * (k % 6))
        k += 1
        t += .86
# opening + end chords on piano
for m in (62, 66, 69, 73, 76):
    place(mus, piano(m, 5, .7), MARKS[0] + (m - 62) * .012, pan=.1)
for j, m in enumerate((74, 78, 81, 85, 88)):
    place(mus, piano(m, 6, .75), E + .5 + j * .14, pan=-.3 + .15 * j)

# bansuri through "Where clouds come home" and "1,900 m"
for t0, m, d in [(6.5, 78, 1.6), (8.2, 81, 1.2), (9.5, 83, 2.4), (13.0, 81, 1.2), (14.3, 78, 1.1), (15.5, 76, 2.4)]:
    place(mus, flute(m, d), t0, pan=.18)
# santoor sparkle on reveals
for t0 in (6.1, 12.5):
    for j, m in enumerate((86, 90, 93, 97)):
        place(mus, santoor(m), t0 + .35 + j * .09, pan=.4 - .25 * j)

# transitions
for mk in MARKS[:4]:
    place(fx, reverse_swell(1.5, .9 if mk > 1 else .6), max(0, mk - 1.5))
    place(fx, boom(.8 if mk > 1 else 1.0), mk)
place(fx, riser(2.0, .8), E - 2.0)
place(fx, boom(1.1), E)
for j, m in enumerate((90, 93, 97)):
    place(fx, chime(m), E + 1.6 + j * .16, pan=.3 - .3 * j)     # logo shine

# air + birds
air = filt(rng.standard_normal(N), 250, 2200)
air *= .006 * (1 + .5 * np.sin(2 * np.pi * np.arange(N) / SR / 8))
fx += np.vstack([air, np.roll(air, 1200)])
t = 1.2
while t < E:
    pan = rng.uniform(-.8, .8)
    for c in range(rng.integers(2, 4)):
        place(fx, chirp(rng.uniform(.5, 1)), t + c * .12, pan=pan)
    t += rng.uniform(1.8, 3.6)


def reverb(x, secs, wet):
    n = int(secs * SR)
    ti = np.arange(n) / SR
    y = np.zeros_like(x)
    for ch in range(2):
        ir = filt(rng.standard_normal(n) * np.exp(-ti * 6.9 / secs), 150, 7000)
        ir /= np.sqrt(np.sum(ir ** 2))
        L = x.shape[1] + n
        y[ch] = np.fft.irfft(np.fft.rfft(x[ch], L) * np.fft.rfft(ir, L), L)[:x.shape[1]]
    return x * (1 - wet) + y * wet * 1.3


mix = reverb(mus, 4.2, .42) + reverb(fx, 2.5, .25)
end = int(DUR * SR)
mix = mix[:, :end]
fi = int(.3 * SR)
mix[:, :fi] *= np.linspace(0, 1, fi)
fo = int(2.2 * SR)
mix[:, -fo:] *= np.linspace(1, 0, fo) ** 1.6
mix *= 10 ** (-17.5 / 20) / np.sqrt(np.mean(mix ** 2))
mix = np.tanh(mix * 1.08) / np.tanh(1.08)
mix *= 0.89 / np.max(np.abs(mix))
with wave.open(str(R / "work" / "score.wav"), "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix.T * 32767).astype(np.int16).tobytes())
print("score ok", DUR)
