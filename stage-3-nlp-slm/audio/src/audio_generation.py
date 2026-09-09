"""Reproducible local Windows speech from approved development text only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'nlp'/'benchmarking'/'src'))
from protocol import ROOT, load_development, split_audit, digest, file_hash, write_json
from audio_preprocessing import load_audio, save_audio, quality
from augmentation import augment

AUDIO=ROOT/'audio'
VOICES=['Microsoft Hazel Desktop','Microsoft David Desktop','Microsoft Zira Desktop']
VARIANTS=['clean_normal','slow_speech','fast_speech','mild_noise','moderate_noise','pause_variation','volume_variation','speaker_variation']

def generate(count=64,seed=42,full=False):
    if full:
        gate=AUDIO/'results'/'pilot_acceptance.json'
        if not gate.exists() or not json.loads(gate.read_text()).get('accepted'):
            raise RuntimeError('Full generation requires measured STT and clinical-preservation pilot approval')
    train,val=load_development('TRAIN'),load_development('VALIDATION')
    split_audit(train,val)
    import importlib.util
    spec=importlib.util.spec_from_file_location('stage3_privacy',ROOT/'data-engineering'/'src'/'de_identification.py')
    privacy=importlib.util.module_from_spec(spec); spec.loader.exec_module(privacy)
    selected=[]
    # Sampling uses metadata, not validation predictions. All source variants inherit split.
    for split,df,n in [('TRAIN',train,count*3//4),('VALIDATION',val,count-count*3//4)]:
        unique=df.drop_duplicates('text')
        for row in unique.sample(n=min(n,len(unique)),random_state=seed).itertuples():
            clean,counts=privacy.scrub_pii(row.text)
            if counts or privacy.verify_no_direct_identifiers(clean):
                raise ValueError('Source needs new privacy/annotation review before audio generation')
            selected.append((split,row))
    jobs=[]; base={}
    for i,(split,row) in enumerate(selected):
        source=digest(str(row.document_id)); folder=AUDIO/'data'/'synthetic'/split.lower(); folder.mkdir(parents=True,exist_ok=True)
        for variant in ['clean_normal','slow_speech','fast_speech','speaker_variation']:
            voice_index=(i+(variant=='speaker_variation'))%len(VOICES)
            audio_id=source[:20]+'_'+variant
            rate=-2 if variant=='slow_speech' else 2 if variant=='fast_speech' else 0
            path=folder/(audio_id+'.wav')
            jobs.append({'audio_id':audio_id,'voice':VOICES[voice_index],'rate':rate,'path':str(path),'text':row.text})
            base[(source,variant)]=(path,voice_index,rate)
    manifest_dir=AUDIO/'data'/'manifests'; manifest_dir.mkdir(parents=True,exist_ok=True)
    jobs_path=manifest_dir/'tts_jobs.json'
    write_json(jobs_path,[j for j in jobs if not Path(j['path']).exists()])
    if json.loads(jobs_path.read_text()):
        subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(Path(__file__).with_name('synthesize.ps1')),'-Jobs',str(jobs_path)],check=True)
    records=[]; hashes=set(); rejected=[]
    for i,(split,row) in enumerate(selected):
        source=digest(str(row.document_id)); clean_path,voice_idx,_=base[(source,'clean_normal')]
        signal,sr=load_audio(clean_path)
        for vi,variant in enumerate(VARIANTS):
            variant_seed=seed+i*len(VARIANTS)+vi
            if (source,variant) in base: path,voice_idx,rate=base[(source,variant)]
            else:
                path=clean_path.with_name(source[:20]+'_'+variant+'.wav'); rate=0; voice_idx=i%len(VOICES)
                save_audio(path,augment(signal,variant,variant_seed),sr)
            qc=quality(path); audio_hash=file_hash(path)
            if not qc['qc_pass'] or audio_hash in hashes:
                rejected.append({'audio_id':path.stem,'reason':'qc_failed' if not qc['qc_pass'] else 'duplicate_audio'})
                continue
            hashes.add(audio_hash)
            labels={e['label'] for e in json.loads(row.ner_entities)}
            import re
            records.append({'audio_id':path.stem,'source_document_hash':source,'source_patient_hash':digest(str(row.patient_id)),
                'split':split,'synthetic_speaker_id':f'speaker_{voice_idx+1:03d}','source_text_hash':digest(row.text),
                'reference_transcript':row.text,'audio_path':str(path.relative_to(ROOT)),'audio_sha256':audio_hash,
                'variant_type':variant,'noise_level':25 if variant=='mild_noise' else 15 if variant=='moderate_noise' else None,
                'speech_rate':rate,'volume_level':.55 if variant=='volume_variation' else 1.,
                'contains_drug':'DRUG_NAME' in labels,'contains_dosage':'DOSAGE' in labels,
                'contains_gene_mutation':'GENE_MUTATION' in labels,'contains_adverse_event':'ADVERSE_EVENT' in labels,
                'contains_negation':bool(re.search(r'\b(no|not|denies|without|negative)\b',row.text,re.I)),
                'generation_seed':variant_seed,**qc})
    df=pd.DataFrame(records)
    df.to_csv(manifest_dir/'dataset_manifest.csv',index=False)
    for split in ['TRAIN','VALIDATION']: df[df.split==split].to_csv(manifest_dir/(split.lower()+'_manifest.csv'),index=False)
    report={'phase':'full' if full else 'pilot','source_documents':len(selected),'files':len(df),'rejected':rejected,
        'speakers':int(df.synthetic_speaker_id.nunique()),'variants':int(df.variant_type.nunique()),
        'hours':float(df.duration_seconds.sum()/3600),'split_counts':df.split.value_counts().to_dict(),
        'qc_pass_rate':len(df)/(len(selected)*len(VARIANTS)),'duplicates_rejected':sum(r['reason']=='duplicate_audio' for r in rejected),
        'clinical_acceptance':'PENDING STT and clinical-preservation evaluation; waveform QC alone is insufficient'}
    write_json(AUDIO/'results'/'audio_generation.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--documents',type=int,default=64); parser.add_argument('--full',action='store_true')
    args=parser.parse_args(); generate(args.documents,full=args.full)
