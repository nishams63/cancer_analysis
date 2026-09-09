import argparse
import json
import sys
import time
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'nlp'/'benchmarking'/'src'))
from protocol import ROOT,write_json,digest,load_development,classification,span_metrics
from stt_model import WhisperSTT, WindowsSTT
from transcript_normalization import normalize_transcript
from evaluation import error_counts,preservation,project_entities

def run(model_name='tiny.en',limit=None):
    audio=ROOT/'audio'; output=audio/'results'/'transcription'/(model_name.replace('/','_')+'.json')
    try: stt=WindowsSTT() if model_name=='windows_sapi' else WhisperSTT(model_name,download_root=str(audio/'models'/'stt'))
    except Exception as exc:
        write_json(audio/'results'/'metrics'/(model_name.replace('/','_')+'.json'),{'status':'NOT RUN','reason':str(exc)})
        raise
    manifest=pd.read_csv(audio/'data'/'manifests'/'validation_manifest.csv')
    if set(manifest.split)!={'VALIDATION'}: raise ValueError('Only validation audio may be scored here')
    if limit: manifest=manifest.head(limit)
    from inference import ClinicalNLPInferenceEngine
    from text_cleaning import clean_clinical_text
    from text_normalization import normalize_clinical_text
    nlp=ClinicalNLPInferenceEngine()
    source={digest(str(r.document_id)):r for r in load_development('VALIDATION').itertuples()}
    records=json.loads(output.read_text()) if output.exists() else []
    done={r['audio_id'] for r in records}
    for m in manifest.itertuples():
        if m.audio_id in done: continue
        ref=source[m.source_document_hash]
        start=time.perf_counter(); transcript=stt.transcribe(ROOT/m.audio_path); stt_seconds=time.perf_counter()-start
        start=time.perf_counter(); normalized=normalize_transcript(transcript); normalize_seconds=time.perf_counter()-start
        start=time.perf_counter(); raw_result=nlp.analyze_document(transcript); result=nlp.analyze_document(normalized); nlp_seconds=(time.perf_counter()-start)/2
        text_result=nlp.analyze_document(ref.text)
        gold=json.loads(ref.ner_entities)
        engine_text=normalize_clinical_text(clean_clinical_text(normalized))
        projected=project_entities(ref.text,engine_text,result['clinical_entities'])
        reference_engine_text=normalize_clinical_text(clean_clinical_text(ref.text))
        text_entities=project_entities(ref.text,reference_engine_text,text_result['clinical_entities'])
        records.append({'audio_id':m.audio_id,'source_document_hash':m.source_document_hash,'variant_type':m.variant_type,
            'transcript':transcript,'normalized_transcript':normalized,'stt_seconds':stt_seconds,'normalization_seconds':normalize_seconds,
            'nlp_seconds':nlp_seconds,'total_seconds':stt_seconds+normalize_seconds+nlp_seconds,
            'real_time_factor':stt_seconds/m.duration_seconds,**error_counts(ref.text,transcript),
            'preservation':preservation(ref.text,transcript,gold),'true_urgency':ref.urgency_level,'true_hazard':ref.hazard_type,
            'text_urgency':text_result['triage_urgency']['predicted_class'],'text_hazard':text_result['toxicity_hazard']['predicted_class'],
            'raw_urgency':raw_result['triage_urgency']['predicted_class'],'audio_urgency':result['triage_urgency']['predicted_class'],
            'audio_hazard':result['toxicity_hazard']['predicted_class'],'gold':gold,'text_entities':text_entities,'audio_entities_projected':projected})
        write_json(output,records); print(model_name,len(records),'/',len(manifest),flush=True)
    def summary(rows):
        result={'files':len(rows),'WER':sum(r['word_errors'] for r in rows)/sum(r['reference_words'] for r in rows),
             'CER':sum(r['character_errors'] for r in rows)/sum(r['reference_characters'] for r in rows)}
        for label in ['DRUG_NAME','DOSAGE','GENE_MUTATION','ADVERSE_EVENT','NUMERIC','NEGATION']:
            n=sum(r['preservation'][label]['total'] for r in rows)
            result[label+'_preservation']=sum(r['preservation'][label]['preserved'] for r in rows)/n if n else None
        for mode in ['text','audio']:
            result[mode+'_urgency']=classification([r['true_urgency'] for r in rows],[r[mode+'_urgency'] for r in rows],['CRITICAL','HIGH','LOW','MEDIUM'])
            result[mode+'_hazard']=classification([r['true_hazard'] for r in rows],[r[mode+'_hazard'] for r in rows],sorted(nlp.baselines.label_manager.hazard_encoder.classes_))
            ents=[r['text_entities' if mode=='text' else 'audio_entities_projected'] for r in rows]
            result[mode+'_ner_exact']=span_metrics([r['gold'] for r in rows],ents)
            result[mode+'_ner_relaxed']=span_metrics([r['gold'] for r in rows],ents,True)
        result['urgency_f1_difference']=result['audio_urgency']['macro_f1']-result['text_urgency']['macro_f1']
        result['latency_seconds']=sum(r['total_seconds'] for r in rows)/len(rows)
        return result
    write_json(audio/'results'/'metrics'/(model_name+'.json'),{'status':'EVALUATED','overall':summary(records),
        'by_condition':{v:summary([r for r in records if r['variant_type']==v]) for v in sorted({r['variant_type'] for r in records})}})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--model',default='tiny.en'); parser.add_argument('--limit',type=int)
    args=parser.parse_args(); run(args.model,args.limit)
