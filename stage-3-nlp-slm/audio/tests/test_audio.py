import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from augmentation import augment
from transcript_normalization import normalize_transcript
from audio_preprocessing import quality,save_audio

def test_normalization_preserves_meaning_tokens():
    assert normalize_transcript(' No\n Cisplatin  75 mg/m2. ')=='No Cisplatin 75 mg/m2.'

def test_deterministic_augmentation():
    x=np.ones(1000,dtype=np.float32)*.2
    assert np.array_equal(augment(x,'mild_noise',42),augment(x,'mild_noise',42))
    assert not np.array_equal(augment(x,'mild_noise',42),augment(x,'mild_noise',43))

def test_silent_audio_rejected(tmp_path):
    path=tmp_path/'silent_test_fixture.wav'
    save_audio(path,np.zeros(16000,dtype=np.float32))
    assert not quality(path)['qc_pass']

def test_no_unknown_augmentation():
    with pytest.raises(ValueError): augment(np.ones(1000),'unknown',42)
