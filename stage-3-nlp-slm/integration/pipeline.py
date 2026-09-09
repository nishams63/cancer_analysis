"""Optional speech entry point reusing existing clinical NLP inference.

Baseline stays selected until the additive validation promotion gate succeeds.
No speech dependencies are imported for text-only inference.
"""
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'nlp'/'src'))

class Stage3Pipeline:
    def __init__(self,stt=None,nlp=None):
        if nlp is None:
            from inference import ClinicalNLPInferenceEngine
            nlp=ClinicalNLPInferenceEngine()
        self.nlp=nlp
        self.stt=stt

    def analyze_text(self,text):
        if not isinstance(text,str) or not text.strip(): raise ValueError('Nonempty text required')
        return {'modality':'TEXT','research_only':True,'prediction':self.nlp.analyze_document(text)}

    def analyze_audio(self,path):
        if self.stt is None: raise RuntimeError('Optional STT backend is not configured')
        sys.path.insert(0,str(ROOT/'audio'/'src'))
        from audio_preprocessing import quality
        from transcript_normalization import normalize_transcript
        start=time.perf_counter(); qc=quality(path)
        if not qc['qc_pass']: raise ValueError('Audio failed quality checks')
        t=time.perf_counter(); transcript=self.stt.transcribe(path); stt_time=time.perf_counter()-t
        normalized=normalize_transcript(transcript)
        prediction=self.analyze_text(normalized)['prediction']
        return {'modality':'AUDIO','research_only':True,'transcript':transcript,
            'normalized_transcript':normalized,'prediction':prediction,'audio_quality':qc,
            'latency_seconds':{'stt':stt_time,'total':time.perf_counter()-start}}
