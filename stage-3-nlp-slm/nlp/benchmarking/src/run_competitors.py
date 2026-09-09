"""Frozen pretrained contextual encoders with genuinely trained task heads.

All encoders share the same development split and head-training protocol.
NER uses per-token contextual representations and a supervised BIO linear head,
never sentence-level embeddings. Missing checkpoints are reported, not simulated.
"""
import argparse
import gc
import json
import time
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, SGDClassifier
from protocol import *

def encode(df,tokenizer,encoder,device):
    from token_alignment import align_spans_to_bio_tokens
    pooled=[]; tokens=[]; targets=[]; offsets=[]
    for i,row in enumerate(df.itertuples()):
        aligned=align_spans_to_bio_tokens(row.text,json.loads(row.ner_entities),tokenizer,max_length=384)
        ids=torch.tensor([aligned['input_ids']],device=device)
        mask=torch.tensor([aligned['attention_mask']],device=device)
        with torch.inference_mode():
            h=encoder(input_ids=ids,attention_mask=mask).last_hidden_state[0].float().cpu().numpy()
        labels=np.array(aligned['labels']); valid=labels!=-100
        pooled.append(h[valid].mean(0)); tokens.append(h[valid].astype(np.float32)); targets.append(labels[valid])
        offsets.append([o for o,v in zip(aligned['offset_mapping'],valid) if v])
        if i%500==0: print('Encoded',i,'/',len(df),flush=True)
    return np.asarray(pooled),tokens,targets,offsets

def run(name,online=False):
    from transformers import AutoTokenizer,AutoModel
    from token_alignment import reconstruct_spans_from_bio_predictions
    config=json.loads((BENCH/'configs'/'benchmark.json').read_text())
    checkpoint=config['models'][name]; resultpath=BENCH/'results'/'validation'/(name+'.json')
    torch.set_num_threads(4); np.random.seed(42); torch.manual_seed(42)
    start=time.perf_counter()
    try:
        tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=not online)
        encoder=AutoModel.from_pretrained(checkpoint,local_files_only=not online)
    except Exception as exc:
        write_json(resultpath,{'name':name,'checkpoint':checkpoint,'status':'NOT RUN','reason':str(exc),'metrics':None})
        print(name,'NOT RUN',str(exc),flush=True); return
    device='cuda' if torch.cuda.is_available() else 'cpu'
    encoder.to(device).eval()
    for param in encoder.parameters(): param.requires_grad_(False)
    params=sum(p.numel() for p in encoder.parameters())
    train,val=load_development('TRAIN'),load_development('VALIDATION'); split_audit(train,val)
    load_seconds=time.perf_counter()-start
    start=time.perf_counter(); x,t,y,off=encode(train,tokenizer,encoder,device)
    training_encode_seconds=time.perf_counter()-start
    start=time.perf_counter(); xv,tv,yv,ov=encode(val,tokenizer,encoder,device)
    validation_encode_seconds=time.perf_counter()-start
    gold=[json.loads(e) for e in val.ner_entities]
    start=time.perf_counter()
    urg=LogisticRegression(class_weight='balanced',max_iter=1500,random_state=42).fit(x,train.urgency_level)
    haz=LogisticRegression(class_weight='balanced',max_iter=1500,random_state=42).fit(x,train.hazard_type)
    token=SGDClassifier(loss='log_loss',random_state=42,average=True,learning_rate='constant',eta0=.002,alpha=.0001)
    best=-1.; wait=0; best_token=None; history=[]
    for epoch in range(15):
        order=np.random.default_rng(42+epoch).permutation(len(t))
        for i in order: token.partial_fit(t[i],y[i],classes=np.arange(9))
        predicted=[reconstruct_spans_from_bio_predictions(text,token.predict(h).tolist(),offsets=o)
                   for text,h,o in zip(val.text,tv,ov)]
        f1=span_metrics(gold,predicted)['micro']['f1']
        history.append({'epoch':epoch+1,'validation_ner_exact_f1':f1})
        print(name,'epoch',epoch+1,'NER exact',f1,flush=True)
        if f1>best:
            import copy
            best=f1; best_token=copy.deepcopy(token); wait=0
        else: wait+=1
        if wait>=3: break
    token=best_token
    train_seconds=time.perf_counter()-start+training_encode_seconds
    start=time.perf_counter()
    pu=urg.predict(xv); ph=haz.predict(xv)
    predicted=[reconstruct_spans_from_bio_predictions(text,token.predict(h).tolist(),offsets=o)
               for text,h,o in zip(val.text,tv,ov)]
    infer_seconds=time.perf_counter()-start+validation_encode_seconds
    out=BENCH/'models'/name; out.mkdir(parents=True,exist_ok=True)
    joblib.dump({'urgency':urg,'hazard':haz,'token':token},out/'heads.joblib')
    write_json(out/'encoder.json',{'checkpoint':checkpoint,'revision':getattr(encoder.config,'_commit_hash',None),
        'protocol':'frozen pretrained encoder, mean-pooled document heads, supervised token BIO head','seed':42})
    result={'name':name,'checkpoint':checkpoint,'status':'EVALUATED','split':'VALIDATION','protocol':'frozen_encoder_trained_heads',
       'device':device,'documents':len(val),'parameter_count':params+sum(m.coef_.size+m.intercept_.size for m in [urg,haz,token]),
       'encoder_parameter_bytes':sum(p.numel()*p.element_size() for p in encoder.parameters()),
       'head_artifact_bytes':(out/'heads.joblib').stat().st_size,'training_seconds':train_seconds,
       'load_seconds':load_seconds,'seconds_per_document':infer_seconds/len(val),
       'urgency':classification(val.urgency_level,pu,list(urg.classes_)),
       'hazard':classification(val.hazard_type,ph,list(haz.classes_)),
       'ner_exact':span_metrics(gold,predicted),'ner_relaxed':span_metrics(gold,predicted,True),'history':history}
    write_json(resultpath,result)
    write_json(BENCH/'results'/'predictions'/(name+'.json'),[{'document_hash':digest(str(row.document_id)),
        'predicted_urgency':str(u),'predicted_hazard':str(h),'entities':e}
        for row,u,h,e in zip(val.itertuples(),pu,ph,predicted)])
    # Hybrid reuses the same contextual representations plus existing structured counts.
    from feature_extraction import extract_structured_concept_features
    from sklearn.preprocessing import StandardScaler
    st=extract_structured_concept_features(train).drop(columns='document_id')
    sv=extract_structured_concept_features(val).drop(columns='document_id')
    scaler=StandardScaler().fit(st); hx=np.hstack([x,scaler.transform(st)]); hv=np.hstack([xv,scaler.transform(sv)])
    hu=LogisticRegression(class_weight='balanced',max_iter=1500,random_state=42).fit(hx,train.urgency_level)
    hh=LogisticRegression(class_weight='balanced',max_iter=1500,random_state=42).fit(hx,train.hazard_type)
    hybrid=dict(result); hybrid['name']=name+'_hybrid'; hybrid['protocol']='frozen_encoder_plus_structured_features'; hybrid['training_seconds']=None; hybrid['seconds_per_document']=None
    hybrid['parameter_count']=params+sum(m.coef_.size+m.intercept_.size for m in [hu,hh,token])
    hybrid['head_artifact_bytes']=None
    hybrid['urgency']=classification(val.urgency_level,hu.predict(hv),list(hu.classes_)); hybrid['hazard']=classification(val.hazard_type,hh.predict(hv),list(hh.classes_))
    write_json(BENCH/'results'/'validation'/(name+'_hybrid.json'),hybrid)
    joblib.dump({'urgency':hu,'hazard':hh,'token':token,'scaler':scaler},out/'hybrid_heads.joblib')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--model',required=True); parser.add_argument('--online',action='store_true')
    args=parser.parse_args(); run(args.model,args.online)
