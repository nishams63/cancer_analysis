"""Read-only frozen baseline evaluation; never calls baseline save/fit/evaluate methods."""
import json
import time
import platform
import joblib
import numpy as np
import pandas as pd
from protocol import *

def run():
    snapshot=BENCH/'results'/'protected_files_before.json'
    if not snapshot.exists(): write_json(snapshot,inventory())
    before=json.loads(snapshot.read_text())
    train,val=load_development('TRAIN'),load_development('VALIDATION')
    audit=split_audit(train,val)
    write_json(BENCH/'results'/'development_audit.json',{
        'train_documents':len(train),'validation_documents':len(val),'overlap':audit,
        'train_patients':int(train.patient_id.nunique()),'validation_patients':int(val.patient_id.nunique()),
        'mean_words':float(train.text.str.split().str.len().mean()),
        'locked_test':'not opened by benchmark; existing integrity tests are separate'})
    from baseline import ClinicalNLPBaselines
    from clinical_concepts import extract_clinical_concepts
    model=ClinicalNLPBaselines.load()
    start=time.perf_counter()
    X,_=model.feature_pipeline.transform(val)
    pu=model.label_manager.inverse_transform_urgency(model.urgency_model.predict(X))
    ph=model.label_manager.inverse_transform_hazard(model.hazard_model.predict(X))
    entities=[extract_clinical_concepts(t,assign_polarity=True) for t in val.text]
    elapsed=time.perf_counter()-start
    gold=[json.loads(e) for e in val.ner_entities]
    result={'name':'baseline_a','status':'EVALUATED','split':'VALIDATION','documents':len(val),
        'urgency':classification(val.urgency_level,pu,list(model.label_manager.urgency_encoder.classes_)),
        'hazard':classification(val.hazard_type,ph,list(model.label_manager.hazard_encoder.classes_)),
        'ner_exact':span_metrics(gold,entities),'ner_relaxed':span_metrics(gold,entities,True),
        'total_inference_seconds':elapsed,'seconds_per_document':elapsed/len(val),
        'documents_per_second':len(val)/elapsed,'training_seconds':None,
        'parameter_count':int(sum(m.coef_.size+m.intercept_.size for m in [model.urgency_model,model.hazard_model])),
        'artifact_bytes':sum(p.stat().st_size for p in (ROOT/'nlp'/'artifacts').rglob('*.joblib')),
        'context_accuracy':None,'context_note':'No corpus context labels asserted; controlled diagnostics separately.',
        'python':platform.python_version()}
    try:
        import psutil
        result['rss_bytes']=psutil.Process().memory_info().rss
    except ImportError: result['rss_bytes']=None
    write_json(BENCH/'results'/'validation'/'baseline_a.json',result)
    predictions=[{'document_hash':digest(str(row.document_id)), 'true_urgency':row.urgency_level,
        'predicted_urgency':u,'true_hazard':row.hazard_type,'predicted_hazard':h,
        'entities':e} for row,u,h,e in zip(val.itertuples(),pu,ph,entities)]
    write_json(BENCH/'results'/'predictions'/'baseline_a.json',predictions)
    after=inventory()
    changed=[p for p,h in before.items() if after.get(p)!=h]
    write_json(BENCH/'results'/'protected_files_check.json',{'changed':changed,'checked_files':len(before)})
    if changed: raise RuntimeError('Protected source files changed')
    print(json.dumps({k:result[k] for k in ['name','documents','seconds_per_document']}),flush=True)
    print('Urgency macro F1',result['urgency']['macro_f1'],'NER exact F1',result['ner_exact']['micro']['f1'],flush=True)

if __name__=='__main__': run()
