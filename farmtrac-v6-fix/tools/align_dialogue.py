import wave, numpy as np, sys
def env(p):
    w=wave.open(p); x=np.frombuffer(w.readframes(w.getnframes()),np.int16).astype(float)/32768
    hop=160; n=len(x)//hop
    e=np.array([np.sqrt(np.mean(x[i*hop:(i+1)*hop]**2)) for i in range(n)])
    d=20*np.log10(e+1e-6); d=np.clip(d-d.max()+40,0,None); return d   # 10ms frames, 40dB range
ref, tgt = env(sys.argv[1]), env(sys.argv[2])
for a,b in [(0.6,1.6),(1.55,2.6)]:
    seg=ref[int(a*100):int(b*100)]; best=None
    for off in range(0, len(tgt)-len(seg)):
        c=np.corrcoef(seg, tgt[off:off+len(seg)])[0,1]
        if best is None or c>best[0]: best=(c,off)
    print(f"v5 [{a},{b}] -> take at {best[1]/100:.2f}s (offset {best[1]/100-a:+.2f}s, r={best[0]:.2f})")
