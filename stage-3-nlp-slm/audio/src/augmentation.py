import numpy as np

def augment(x,variant,seed):
    rng=np.random.default_rng(seed)
    if variant in ('mild_noise','moderate_noise'):
        snr=25 if variant=='mild_noise' else 15
        noise=rng.normal(size=len(x))
        noise*=np.sqrt(np.mean(x*x))/(10**(snr/20)*np.sqrt(np.mean(noise*noise)))
        y=x+noise
        return y/max(1.,np.max(np.abs(y))/.98)
    if variant=='volume_variation': return x*.55
    if variant=='pause_variation':
        # Add silence only at observed quiet runs, never remove speech samples.
        quiet=np.flatnonzero(np.abs(x)<.001)
        candidates=quiet[(quiet>1600)&(quiet<len(x)-1600)]
        points=sorted(rng.choice(candidates,min(3,len(candidates)),replace=False)) if len(candidates) else []
        chunks=[]; last=0
        for point in points:
            chunks.extend([x[last:point],np.zeros(1600,dtype=x.dtype)]); last=point
        chunks.append(x[last:]); return np.concatenate(chunks)
    raise ValueError('Variant must be synthesized by TTS or explicitly supported')
