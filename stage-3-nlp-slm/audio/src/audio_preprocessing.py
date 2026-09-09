"""Audio QC of genuine speech; does not create placeholder signals."""
import wave
import numpy as np

def load_audio(path):
    with wave.open(str(path),'rb') as f:
        sr,channels,width=f.getframerate(),f.getnchannels(),f.getsampwidth()
        if width!=2 or channels!=1 or sr!=16000: raise ValueError('Expected 16 kHz mono PCM16')
        data=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768.
    return data,sr

def quality(path):
    x,sr=load_audio(path)
    if len(x)==0: raise ValueError('Empty audio')
    duration=len(x)/sr
    rms=float(np.sqrt(np.mean(x*x)))
    peak=float(np.max(np.abs(x)))
    clipping=float(np.mean(np.abs(x)>=.999))
    silence=float(np.mean(np.abs(x)<.001))
    passed=duration>=1 and rms>.001 and peak>0 and clipping<.01 and silence<.95
    return {'sample_rate':sr,'channels':1,'duration_seconds':duration,'rms':rms,
            'peak_amplitude':peak,'clipping_ratio':clipping,'silence_ratio':silence,'qc_pass':bool(passed)}

def save_audio(path,x,sr=16000):
    if np.max(np.abs(x))>1: raise ValueError('Signal would clip')
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes((x*32767).astype('<i2').tobytes())
