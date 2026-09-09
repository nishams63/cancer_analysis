from pathlib import Path
import sys
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'nlp'/'benchmarking'/'src'))
from protocol import ROOT,file_hash

def test_generated_manifest_integrity():
    path=ROOT/'audio'/'data'/'manifests'/'dataset_manifest.csv'
    if not path.exists(): pytest.skip('Generate real pilot before integration QC')
    df=pd.read_csv(path)
    assert set(df.split)=={'TRAIN','VALIDATION'}
    assert not df.audio_sha256.duplicated().any()
    assert df.groupby('source_document_hash').split.nunique().max()==1
    assert df.groupby('source_patient_hash').split.nunique().max()==1
    assert df.groupby('source_text_hash').split.nunique().max()==1
    assert df.qc_pass.all() and (df.sample_rate==16000).all()
    assert not {'patient_id','encounter_id','document_id'} & set(df.columns)
    for row in df.itertuples():
        assert file_hash(ROOT/row.audio_path)==row.audio_sha256

def test_reference_aware_metrics():
    sys.path.insert(0,str(ROOT/'audio'/'src'))
    from evaluation import error_counts,project_entities
    assert error_counts('no fever','fever')['word_errors']==1
    assert error_counts('no fever','no fever')['word_errors']==0
    e={'start':0,'end':5,'label':'ADVERSE_EVENT'}
    assert project_entities('no fever','fever',[e])[0]['start']==3

def test_text_pipeline_without_speech_dependency():
    sys.path.insert(0,str(ROOT/'integration'))
    from pipeline import Stage3Pipeline
    class Dummy:
        def analyze_document(self,text): return {'text':text}
    service=Stage3Pipeline(nlp=Dummy())
    assert service.analyze_text('No fever')['prediction']['text']=='No fever'
    with pytest.raises(RuntimeError,match='not configured'): service.analyze_audio('unused.wav')
