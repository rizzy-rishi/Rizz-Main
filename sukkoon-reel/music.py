"""Synthesise the BGM + SFX track for the Sukkoon reel (40 s, 44.1 kHz stereo WAV).

Everything is generated from code (no samples): a santoor-like plucked arpeggio,
a bansuri-style flute, warm pads, soft percussion, and SFX (whooshes, bird calls,
map pops, SOLD stamps, impacts, bell). SFX times mirror the timeline in index.html.
"""
import pathlib
import wave

import numpy as np

SR = 44100
DUR = 40.0
N = int(SR * (DUR + 0.5))
rng = np.random.default_rng(7)
OUT = pathlib.Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

BPM = 96
BEAT = 60 / BPM          # 0.625 s
BAR = 4 * BEAT           # 2.5 s
T0 = 3.0                 # music grid origin (scene 2 start)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def buf():
    return np.zeros((2, N))


def place(bus, sig, t, pan=0.0, gain=1.0):
    """Add mono (or stereo) signal to bus at time t with constant-power pan."""
    i = int(t * SR)
    if i >= N:
        return
    if sig.ndim == 1:
        l = np.cos((pan + 1) * np.pi / 4)
        r = np.sin((pan + 1) * np.pi / 4)
        sig = np.vstack([sig * l, sig * r])
    n = min(sig.shape[1], N - i)
    bus[:, i:i + n] += gain * sig[:, :n]


def tt(d):
    return np.arange(int(d * SR)) / SR


def adsr(n, a=0.01, d=0.1, s=0.7, r=0.2):
    e = np.ones(n) * s
    ia, idd, ir = int(a * SR), int(d * SR), int(r * SR)
    ia = min(ia, n)
    e[:ia] = np.linspace(0, 1, ia, endpoint=False)
    j = min(ia + idd, n)
    e[ia:j] = np.linspace(1, s, j - ia, endpoint=False)
    if ir > 0 and n > ir:
        e[-ir:] *= np.linspace(1, 0, ir)
    return e


def fft_filter(x, lo=None, hi=None):
    """Brick-ish band filter with soft edges (static)."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    g = np.ones_like(f)
    if hi:
        g *= 1 / (1 + (f / hi) ** 4)
    if lo:
        g *= 1 / (1 + (lo / np.maximum(f, 1e-3)) ** 4)
    return np.fft.irfft(X * g, len(x))


def noise(d):
    return rng.standard_normal(int(d * SR))


# ------------------------------------------------------------------ instruments
def pad_note(m, d, bright=4.0):
    t = tt(d)
    out = np.zeros_like(t)
    for det in (-0.07, 0.0, 0.07):
        f = mtof(m + det)
        for k in range(1, 9):
            out += np.sin(2 * np.pi * f * k * t + rng.random() * 6.28) / k * np.exp(-k / bright)
    return out * adsr(len(t), a=0.9, d=0.5, s=0.85, r=1.0) * 0.08


def pluck(m, d=1.6, bright=0.5):
    """Vectorised Karplus-Strong (santoor-ish)."""
    f = mtof(m)
    p = max(2, int(round(SR / f)))
    n = int(d * SR)
    y = np.zeros(n + p + 1)
    y[:p] = fft_filter(rng.uniform(-1, 1, p * 4), hi=SR * bright / 4)[:p]
    decay = 0.996
    for i in range(p, n + p, p):
        seg = y[i - p:i - p + p + 1]
        blk = 0.5 * (seg[:-1] + seg[1:]) * decay
        y[i:i + p] = blk[:len(y[i:i + p])]
    out = y[:n]
    out *= adsr(n, a=0.002, d=0.05, s=1.0, r=0.3)
    return out * 0.35


def flute(m, d):
    t = tt(d)
    f = mtof(m)
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * t) * np.clip((t - 0.18) * 4, 0, 1)
    ph = 2 * np.pi * np.cumsum(f * vib) / SR
    tone = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
    breath = fft_filter(noise(d), lo=1500, hi=6000) * 0.05
    env = adsr(len(t), a=0.09, d=0.15, s=0.8, r=min(0.3, d * 0.4))
    return (tone + breath) * env * 0.12


def bass(m, d):
    t = tt(d)
    f = mtof(m)
    s = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)
    return s * adsr(len(t), a=0.005, d=0.12, s=0.5, r=0.08) * 0.35


def kick(g=1.0):
    t = tt(0.4)
    f = 45 + 90 * np.exp(-t * 28)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)
    click = np.exp(-t * 300) * 0.3
    return (s + click) * 0.8 * g


def clap(g=1.0):
    n = fft_filter(noise(0.25), lo=900, hi=5000)
    t = tt(0.25)
    env = np.exp(-t * 22) * (1 + 0.6 * (np.sin(2 * np.pi * 90 * t) > 0) * (t < 0.03))
    return n * env * 0.25 * g


def hat(g=1.0):
    n = fft_filter(noise(0.06), lo=7000)
    return n * np.exp(-tt(0.06) * 70) * 0.12 * g


def tick(g=1.0, f=2400):
    t = tt(0.05)
    return (np.sin(2 * np.pi * f * t) * np.exp(-t * 160) + fft_filter(noise(0.05), lo=3000) * np.exp(-t * 250) * 0.4) * 0.22 * g


def impact(g=1.0):
    t = tt(2.5)
    boom = np.sin(2 * np.pi * np.cumsum(38 + 60 * np.exp(-t * 6)) / SR) * np.exp(-t * 1.6)
    crack = fft_filter(noise(2.5), hi=3500) * np.exp(-t * 7) * 0.5
    return (boom + crack) * 0.9 * g


def whoosh(d=0.6, g=1.0):
    n = noise(d)
    out = np.zeros_like(n)
    seg = 8
    L = len(n) // seg
    for k in range(seg):  # stepped band sweep up then down
        c = 500 + 3500 * np.sin(np.pi * (k + 0.5) / seg)
        out[k * L:(k + 1) * L] = fft_filter(n[k * L:(k + 1) * L], lo=c * 0.5, hi=c * 1.6)
    env = np.sin(np.pi * np.linspace(0, 1, len(n))) ** 2
    return out * env * 0.5 * g


def riser(d=1.2, g=1.0):
    t = tt(d)
    sw = np.sin(2 * np.pi * np.cumsum(200 + 1400 * (t / d) ** 2) / SR) * 0.25
    n = fft_filter(noise(d), lo=800) * 0.3
    return (sw + n) * (t / d) ** 2 * g


def chirp(g=1.0):
    d = 0.12
    t = tt(d)
    f0 = rng.uniform(2800, 4200)
    f = f0 + rng.uniform(800, 1800) * np.sin(np.pi * t / d) * rng.choice([-1, 1])
    s = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return s * np.sin(np.pi * t / d) ** 2 * 0.08 * g


def blip(f=880, g=1.0):
    t = tt(0.25)
    fr = f * (1 + 0.5 * (1 - np.exp(-t * 40)))
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t * 18) * 0.25 * g


def stamp(g=1.0):
    t = tt(0.22)
    thud = np.sin(2 * np.pi * np.cumsum(70 + 140 * np.exp(-t * 60)) / SR) * np.exp(-t * 22)
    snap = fft_filter(noise(0.22), lo=1200, hi=6000) * np.exp(-t * 90) * 0.6
    return (thud + snap) * 0.45 * g


def bell(m=81, d=3.0, g=1.0):
    t = tt(d)
    fc = mtof(m)
    mod = np.sin(2 * np.pi * fc * 3.5 * t) * 2.2 * np.exp(-t * 2.5)
    return np.sin(2 * np.pi * fc * t + mod) * np.exp(-t * 1.6) * 0.22 * g


def reverb(x, secs=2.4, wet=0.3):
    n = int(secs * SR)
    t = np.arange(n) / SR
    out = np.zeros_like(x)
    for ch in range(2):
        ir = rng.standard_normal(n) * np.exp(-t * 6.9 / secs)
        ir = fft_filter(ir, hi=6000)
        ir /= np.sqrt(np.sum(ir ** 2))
        L = x.shape[1] + n
        y = np.fft.irfft(np.fft.rfft(x[ch], L) * np.fft.rfft(ir, L), L)[:x.shape[1]]
        out[ch] = y
    return x * (1 - wet) + out * wet * 1.4


# ------------------------------------------------------------------ arrangement
music, drums, sfx = buf(), buf(), buf()

CHORDS = {  # root-position voicings (MIDI)
    "D": [50, 54, 57, 62], "Bm": [47, 50, 54, 59], "G": [43, 47, 50, 55], "A": [45, 49, 52, 57],
}
PROG = ["D", "Bm", "G", "A"]

# -- intro tension 0 - 3.0: low drone
d = tt(3.2)
drone = (np.sin(2 * np.pi * mtof(38) * d) + 0.5 * np.sin(2 * np.pi * mtof(45) * d)) * np.clip(d / 0.4, 0, 1) * 0.18
place(music, drone * np.linspace(1, 0.2, len(d)), 0.0)

# -- pads + arpeggio from 3.0 to 34.55
bar_i = 0
t = T0
while t < 34.5:
    ch = CHORDS[PROG[bar_i % 4]]
    length = min(BAR + 1.0, 34.55 - t + 0.6)
    for m in ch:
        place(music, pad_note(m, length), t, pan=rng.uniform(-0.4, 0.4))
    # santoor arpeggio (8ths)
    pat = [ch[0] + 24, ch[2] + 12, ch[1] + 24, ch[2] + 12, ch[3] + 12, ch[2] + 12, ch[1] + 24, ch[2] + 24]
    for k, m in enumerate(pat):
        st = t + k * BEAT / 2
        if st >= 34.5:
            break
        g = 0.8 if k % 2 == 0 else 0.55
        if t >= 26.2:
            g *= 1.15
        place(music, pluck(m, 1.4), st, pan=-0.35 + 0.1 * (k % 8), gain=g)
    # bass: sustained root until urgency, then driving 8ths
    if t < 21.5:
        place(music, bass(ch[0] - 12, BAR) * 0.7, t)
    else:
        for k in range(8):
            st = t + k * BEAT / 2
            if st < 34.5:
                place(music, bass(ch[0] - 12, BEAT / 2 * 0.9), st, gain=0.9)
    bar_i += 1
    t += BAR

# -- bansuri melody (D major pentatonic)
MEL = [(3.35, 78, .55), (3.95, 81, .95), (5.0, 83, .4), (5.4, 81, 1.0),
       (6.6, 78, .5), (7.2, 76, .6), (7.85, 74, 1.3),
       (9.4, 74, .3), (9.7, 76, .3), (10.0, 78, .8), (10.9, 76, 1.0),
       (12.0, 81, .5), (12.6, 83, .4), (13.0, 81, 1.4),
       (15.3, 78, .4), (15.7, 76, .4), (16.1, 74, 1.1),
       (17.5, 81, .6), (18.2, 78, 1.1), (19.6, 76, .5), (20.2, 74, 1.3)]
for st, m, dd in MEL:
    place(music, flute(m, dd + 0.25), st, pan=0.15)

# -- ending: impact pad + slow descending santoor + bell
for m in [50, 57, 62, 64, 66]:
    place(music, pad_note(m, 5.4, bright=5) * 1.2, 34.55, pan=rng.uniform(-0.5, 0.5))
for k, m in enumerate([86, 81, 78, 76, 74, 69]):
    place(music, pluck(m, 2.5), 35.0 + k * 0.42, pan=-0.3 + 0.12 * k, gain=0.7)
place(music, bass(38, 4.5) * 1.2, 34.55)

# -- percussion
def beat_times(a, b, step):
    x = T0 + np.ceil((a - T0) / step - 1e-9) * step
    out = []
    while x < b - 1e-6:
        out.append(float(x))
        x += step
    return out

for x in beat_times(8.2, 21.55, BEAT * 2):
    place(drums, kick(0.6), x)
for x in beat_times(8.2, 21.55, BEAT / 2):
    place(drums, hat(0.6), x, pan=0.3)
for x in beat_times(21.55, 34.4, BEAT):
    place(drums, kick(0.9), x)
for x in beat_times(21.55 + BEAT, 34.4, BEAT * 2):
    place(drums, clap(0.8), x, pan=-0.1)
for x in beat_times(21.55, 26.2, BEAT / 2):
    place(drums, hat(0.8), x, pan=0.3)

# clock ticks: hook + urgency
for k, x in enumerate(np.arange(0.9, 3.0, BEAT / 2)):
    place(sfx, tick(0.8, 2400 if k % 2 == 0 else 1900), x, pan=0.2)
for k, x in enumerate(beat_times(26.2, 31.3, BEAT / 2)):
    place(sfx, tick(0.75, 2400 if k % 2 == 0 else 1900), x, pan=0.2)
for k, x in enumerate(beat_times(31.3, 34.4, BEAT / 4)):
    place(sfx, tick(0.8, 2600 if k % 2 == 0 else 2000), x, pan=0.2)

# -- SFX
place(sfx, whoosh(0.5, 0.6), 0.0)
place(sfx, riser(0.4, 0.6), 0.05)
place(sfx, impact(1.0), 0.45)
place(sfx, blip(1320, 0.8), 1.6, pan=0.2)
for x in [2.9, 8.0, 10.4, 12.8, 15.05, 21.4, 26.05]:
    place(sfx, whoosh(0.55, 0.9), x, pan=0.0)
for w in (3.6, 8.0), (15.6, 20.8):  # birds
    x = w[0]
    while x < w[1]:
        pan = rng.uniform(-0.8, 0.8)
        for c in range(rng.integers(2, 5)):
            place(sfx, chirp(rng.uniform(0.6, 1.0)), x + c * 0.14, pan=pan)
        x += rng.uniform(0.7, 1.4)
place(sfx, blip(660, 0.9), 15.8)
for i in range(4):
    place(sfx, blip(880 + 110 * i, 0.8), 16.8 + i * 0.75, pan=[0.5, 0.5, -0.5, -0.5][i])
place(sfx, whoosh(0.4, 0.7), 22.35, pan=-0.6)
place(sfx, whoosh(0.4, 0.7), 22.75, pan=0.6)
for i in range(3):
    place(sfx, blip(990, 0.6), 23.6 + i * 0.45)
n_sold = 23 - 3
for i in range(n_sold):  # mirrors soldTimes() in index.html
    place(sfx, stamp(1.0), 27.0 + 4.0 * np.sqrt(i / (n_sold - 1)), pan=rng.uniform(-0.3, 0.3))
place(sfx, impact(0.8), 31.3)
place(sfx, riser(1.15, 0.9), 33.4)
place(sfx, impact(1.1), 34.55)
place(sfx, bell(81, 3.5, 1.0), 34.95, pan=0.1)
place(sfx, bell(86, 3.0, 0.6), 35.25, pan=-0.2)
place(sfx, blip(1175, 0.7), 36.0)

# ------------------------------------------------------------------ mix
music = reverb(music, 2.8, 0.35)
drums = reverb(drums, 1.2, 0.12)
sfx = reverb(sfx, 1.8, 0.2)
mix = music * 1.0 + drums * 0.9 + sfx * 1.0

# fade in/out
fi = int(0.05 * SR)
mix[:, :fi] *= np.linspace(0, 1, fi)
fo0, fo1 = int(38.2 * SR), int(DUR * SR)
mix[:, fo0:fo1] *= np.linspace(1, 0, fo1 - fo0)
mix[:, fo1:] = 0
mix = mix[:, :int(DUR * SR)]

rms = np.sqrt(np.mean(mix ** 2))
mix *= 10 ** (-15 / 20) / rms          # ~ -15 dBFS RMS
mix = np.tanh(mix * 1.1) / np.tanh(1.1)  # soft limiter
mix *= 0.93 / np.max(np.abs(mix))

pcm = (mix.T * 32767).astype(np.int16)
with wave.open(str(OUT / "music.wav"), "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote", OUT / "music.wav", "rms dBFS", 20 * np.log10(np.sqrt(np.mean(mix ** 2))))
