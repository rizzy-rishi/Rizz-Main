"""Dreamy ambient bed for the Pangot valley reel (20.2 s): pads, santoor, bansuri, air, birds."""
import pathlib
import wave

import numpy as np

SR, DUR = 44100, 20.2
N = int(SR * (DUR + 3))
rng = np.random.default_rng(3)
OUT = pathlib.Path(__file__).parent / "out"


def mtof(m):
    return 440 * 2 ** ((m - 69) / 12)


def tt(d):
    return np.arange(int(d * SR)) / SR


def filt(x, lo=None, hi=None):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    g = np.ones_like(f)
    if hi:
        g /= 1 + (f / hi) ** 4
    if lo:
        g /= 1 + (lo / np.maximum(f, 1e-3)) ** 4
    return np.fft.irfft(X * g, len(x))


def env(n, a, r):
    e = np.ones(n)
    ia, ir = min(int(a * SR), n), min(int(r * SR), n)
    e[:ia] = np.linspace(0, 1, ia) ** 2
    e[n - ir:] *= np.linspace(1, 0, ir) ** 2
    return e


bus = np.zeros((2, N))


def place(sig, t, pan=0.0, g=1.0):
    i = int(t * SR)
    if sig.ndim == 1:
        sig = np.vstack([sig * np.cos((pan + 1) * np.pi / 4), sig * np.sin((pan + 1) * np.pi / 4)])
    n = min(sig.shape[1], N - i)
    bus[:, i:i + n] += g * sig[:, :n]


def pad(m, d):
    t = tt(d)
    out = sum(np.sin(2 * np.pi * mtof(m + det) * k * t + rng.random() * 6) / k * np.exp(-k / 2.5)
              for det in (-.06, 0, .06) for k in range(1, 6))
    return out * env(len(t), 1.8, 2.2) * 0.05


def pluck(m, d=3.0):
    t = tt(d)
    f = mtof(m)
    s = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * (1.6 + 1.4 * r)) for r, a in ((1, 1), (2, .35), (3, .12), (4.1, .05)))
    return s * env(len(t), .004, .4) * 0.11


def flute(m, d):
    t = tt(d)
    f = mtof(m) * (1 + .006 * np.sin(2 * np.pi * 5 * t) * np.clip((t - .25) * 3, 0, 1))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) + .15 * np.sin(2 * ph) + filt(rng.standard_normal(len(t)), 1500, 5000) * .05
    return s * env(len(t), .25, min(.6, d * .5)) * 0.07


def chirp(g=1.0):
    t = tt(.1)
    f = rng.uniform(3000, 4300) + rng.uniform(600, 1400) * np.sin(np.pi * t / .1) * rng.choice([-1, 1])
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / .1) ** 2 * .025 * g


# chords: Dmaj9, Bm11, Gmaj7, Aadd9 (5 s each)
CH = [[50, 54, 57, 61, 64], [47, 54, 57, 62, 64], [43, 50, 54, 57, 62], [45, 52, 57, 59, 64]]
for i, ch in enumerate(CH):
    for m in ch:
        place(pad(m, 6.6), i * 5.0, pan=rng.uniform(-.6, .6))
    place(pad(ch[0] - 12, 6.6) * .8, i * 5.0)
# sparse santoor-like plucks
for i, ch in enumerate(CH):
    for k, (dt, idx) in enumerate([(0.6, 4), (1.6, 2), (2.4, 3), (3.6, 1), (4.3, 4)]):
        if i * 5 + dt > 19.4:
            break
        place(pluck(ch[idx] + 12, 3.2), i * 5 + dt, pan=-.4 + .2 * k)
# bansuri phrase in the middle
for t0, m, d in [(6.2, 78, 1.4), (7.7, 81, 1.1), (8.9, 83, 2.2), (12.2, 81, 1.0), (13.3, 78, 1.0), (14.4, 76, 2.6)]:
    place(flute(m, d), t0, pan=.15)
# air / wind bed
air = filt(rng.standard_normal(N), 200, 1800)
air *= 0.012 * (1 + .5 * np.sin(2 * np.pi * np.arange(N) / SR / 7))
bus += np.vstack([air, np.roll(air, 900)])
# distant birds
t = 1.5
while t < 17:
    pan = rng.uniform(-.8, .8)
    for c in range(rng.integers(2, 4)):
        place(chirp(rng.uniform(.5, 1)), t + c * .13, pan=pan)
    t += rng.uniform(1.6, 3.2)
# final shimmer on the end card
place(pluck(86, 3.5) * .8, 17.1, pan=.2)
place(pluck(90, 3.5) * .6, 17.35, pan=-.2)

# long reverb
n = int(3.5 * SR)
ir_t = np.arange(n) / SR
wet = np.zeros_like(bus)
for ch in range(2):
    ir = filt(rng.standard_normal(n) * np.exp(-ir_t * 6.9 / 3.5), hi=6000)
    ir /= np.sqrt(np.sum(ir ** 2))
    L = N + n
    wet[ch] = np.fft.irfft(np.fft.rfft(bus[ch], L) * np.fft.rfft(ir, L), L)[:N]
mix = bus * .6 + wet * .7
end = int(DUR * SR)
mix = mix[:, :end]
fi = int(1.5 * SR)
mix[:, :fi] *= np.linspace(0, 1, fi) ** 2
fo = int(2.8 * SR)
mix[:, -fo:] *= np.linspace(1, 0, fo) ** 2
mix *= 10 ** (-19 / 20) / np.sqrt(np.mean(mix ** 2))
mix = np.tanh(mix * 1.05)
mix *= 0.89 / np.max(np.abs(mix))
with wave.open(str(OUT / "music.wav"), "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix.T * 32767).astype(np.int16).tobytes())
print("ok")
