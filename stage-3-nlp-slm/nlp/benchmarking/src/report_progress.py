"""Generate auditable progress reports only from persisted measured results."""
import json
import pandas as pd
from protocol import ROOT,BENCH,write_json,inventory

def markdown_table(df):
    df=df.fillna('NA')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'

def run():
    docs=[]
    for p in sorted((BENCH/'results'/'validation').glob('*.json')):
        d=json.loads(p.read_text()); docs.append(d)
    rows=[]
    for d in docs:
        rows.append({'model':d['name'],'status':d['status'],
            'urgency_macro_f1':d.get('urgency',{}).get('macro_f1'),
            'critical_recall':d.get('urgency',{}).get('critical_recall'),
            'hazard_macro_f1':d.get('hazard',{}).get('macro_f1'),
            'ner_exact_f1':d.get('ner_exact',{}).get('micro',{}).get('f1'),
            'ner_relaxed_f1':d.get('ner_relaxed',{}).get('micro',{}).get('f1'),
            'seconds_per_document':d.get('seconds_per_document'),'training_seconds':d.get('training_seconds'),
            'parameters':d.get('parameter_count'),'reason':d.get('reason','')})
    df=pd.DataFrame(rows)
    dest=BENCH/'results'/'comparisons'; dest.mkdir(parents=True,exist_ok=True)
    df.to_csv(dest/'model_benchmark.csv',index=False,na_rep='NA')
    report=BENCH/'reports'; report.mkdir(parents=True,exist_ok=True)
    intro='# Stage 3 development benchmark - progress, not final promotion\n\nAll scores below are fresh VALIDATION measurements. No locked-test metric is used for selection. Missing values mean unmeasured, not zero.\n\n'
    columns={
        'model_comparison':['model','status','urgency_macro_f1','critical_recall','hazard_macro_f1','ner_exact_f1'],
        'ner_comparison':['model','status','ner_exact_f1','ner_relaxed_f1'],
        'urgency_comparison':['model','status','urgency_macro_f1','critical_recall'],
        'hazard_comparison':['model','status','hazard_macro_f1'],
        'efficiency_comparison':['model','status','seconds_per_document','training_seconds','parameters']}
    for name,cols in columns.items():
        (report/(name+'.md')).write_text(intro+markdown_table(df[cols]),encoding='utf-8')
    protocol='''# Preregistered development protocol

TRAIN fits all task heads and preprocessing. VALIDATION selects checkpoints.
Locked-test content is not opened by these benchmark runners. Existing regression
tests may inspect locked-test schema/isolation but no prediction tuning uses it.
Data guards enforce patient, encounter, document, canonical-text and temporal isolation.

Frozen contextual encoders share max length 384, seed 42, class-weighted logistic
document heads and a supervised BIO token head. Token head: up to 15 epochs,
validation exact-span F1 checkpoint selection, patience 3. This is not full
Transformer fine-tuning. Raw text offsets are never applied after whitespace collapse.

Primary objectives: urgency macro F1 and exact NER F1. Critical recall may not fall
more than 0.01 absolute below fresh baseline validation recall. No final promotion
before robustness, memory, latency and reproducibility evidence is complete.
Context-corpus accuracy is NA without independently annotated polarity labels.
'''
    (report/'benchmark_protocol.md').write_text(protocol,encoding='utf-8')
    (report/'model_selection.md').write_text('# BASELINE RETAINED - provisional\n\nNo production/pipeline promotion has been authorized by a completed benchmark. Task-head experiments and speech dependencies may still be in progress. This is not a claim that all competitors lost.\n',encoding='utf-8')
    (report/'robustness_comparison.md').write_text('# Robustness status\n\nNOT COMPLETE. Whitespace offset invariants and audio variants have tests; a complete paired NLP robustness benchmark is not yet measured. No robustness score is fabricated.\n',encoding='utf-8')
    baseline=next((d for d in docs if d['name']=='baseline_a'),None)
    if baseline:
        entity_rows=[{'entity':k,**v} for k,v in baseline['ner_exact']['per_type'].items()]
        error='# Measured baseline validation extraction errors\n\n'+markdown_table(pd.DataFrame(entity_rows))
        error+='\nFalse negatives = gold - tp. False positives = predicted - tp. Classification confusion matrices and per-class support are in results/validation/baseline_a.json. Causal explanations are not established by these counts.\n'
        (report/'error_analysis.md').write_text(error,encoding='utf-8')
    (report/'final_benchmark_report.md').write_text(intro+markdown_table(df[['model','status','urgency_macro_f1','ner_exact_f1']])+'\nINCOMPLETE: final cross-model selection and complete efficiency/robustness evaluation remain outstanding.\n',encoding='utf-8')
    audio=ROOT/'audio'; ar=audio/'reports'; ar.mkdir(parents=True,exist_ok=True)
    generation=audio/'results'/'audio_generation.json'
    if generation.exists():
        g=json.loads(generation.read_text())
        text='# Real synthetic speech pilot\n\n'+markdown_table(pd.DataFrame([{'measurement':k,'value':str(v)} for k,v in g.items()]))
        text+='\nThree voices actually synthesized; other registered voices could not run. Waveform QC is not transcription or clinical quality. Full generation remains gated. Source notes are synthetic and do not represent real patient speech.\n'
        for name in ['audio_data_report','audio_data_quality_report']:(ar/(name+'.md')).write_text(text,encoding='utf-8')
    stt=[]
    for p in (audio/'results'/'metrics').glob('*.json'):
        d=json.loads(p.read_text()); overall=d.get('overall',{})
        stt.append({'model':p.stem,'status':d.get('status'),'WER':overall.get('WER'),'CER':overall.get('CER'),
            'files':overall.get('files'),'latency_seconds':overall.get('latency_seconds'),'reason':d.get('reason','')})
    sttdf=pd.DataFrame(stt,columns=['model','status','WER','CER','files','latency_seconds','reason'])
    ac=audio/'results'/'comparisons'; ac.mkdir(parents=True,exist_ok=True)
    sttdf.to_csv(ac/'audio_model_benchmark.csv',index=False,na_rep='NA')
    for name in ['stt_model_comparison','transcription_evaluation','audio_robustness','downstream_impact','error_analysis','final_audio_report']:
        (ar/(name+'.md')).write_text('# Audio benchmark progress\n\n'+markdown_table(sttdf)+'\nINCOMPLETE: full pilot clinical acceptance, full dataset generation, selected-model downstream comparison and speech adaptation are not claimed complete. Per-condition and paired downstream values, when measured, reside in results/metrics.\n',encoding='utf-8')
    final=ROOT/'reports'; final.mkdir(parents=True,exist_ok=True)
    result=ROOT/'results'; result.mkdir(parents=True,exist_ok=True)
    pd.concat([df,sttdf],ignore_index=True).to_csv(result/'final_model_comparison.csv',index=False,na_rep='NA')
    before=json.loads((BENCH/'results'/'protected_files_before.json').read_text()); after=inventory()
    changed=[p for p,h in before.items() if after.get(p)!=h]
    write_json(BENCH/'results'/'protected_files_check.json',{'checked_files':len(before),'changed':changed})
    (final/'final_stage3_upgrade_report.md').write_text('# Stage 3 upgrade - work in progress\n\n'+
        'Implemented additive development guards, validation baseline scoring, contextual task-head competitor runner, real local TTS pilot, audio QC/manifests, optional STT adapters, paired downstream metrics and text/audio integration.\n\n'+
        markdown_table(df[['model','status','urgency_macro_f1','critical_recall','hazard_macro_f1','ner_exact_f1']])+ '\n'+markdown_table(sttdf)+
        f'\nProtected files verified: {len(before)}; changes: {changed}.\n\n'+
        'BASELINE RETAINED provisionally. This is not a completed final decision. GPU enablement and Whisper dependency downloads have encountered repeated network/DNS failures. Complete competitor training, STT pilot acceptance, full audio generation, robustness/efficiency and final promotion remain outstanding. No clinical claims.\n',encoding='utf-8')
    print(df[['model','status','urgency_macro_f1','ner_exact_f1']].to_string(index=False),flush=True)

if __name__=='__main__': run()
