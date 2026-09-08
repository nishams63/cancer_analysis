"""Reproducible experiments with prediction-level evidence and explicit provenance.

No expected winner, no manufactured missing metrics, and no test-set selection.
"""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, mean_absolute_error, mean_squared_error, r2_score)

from dl import config, dataset_v2
from dl.research_data import AuditedTemporalDataset, audit_manifests, composition_bags, stable_seed
from dl.models.research_models import PretrainedPathology, CachedMIL
from dl.models.temporal_transformer import ContinuousTemporalTransformer
from dl.models.multimodal_fusion import build_fusion_model, FixedWeightedFusion
from dl.src.temporal_model import BiLSTMForecaster
from dl.uncertainty_calibration import MCDropoutEvaluator, TemperatureScaler, compute_expected_calibration_error, compute_brier_score
from dl.ood_detector import LatentOODDetector

VERSION = 'validated-v3-landmark90-composition50'
SPLITS = ('train','validation','test')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def safe_json(value):
    if isinstance(value, dict):
        return {str(k):safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(safe_json(value),indent=2,allow_nan=False),encoding='utf-8')


def classification(y, probabilities):
    y,p = np.asarray(y),np.asarray(probabilities)
    if not len(y) or not np.isfinite(p).all():
        raise ValueError('Empty cohort or invalid predictions')
    multiclass = p.ndim == 2
    pred = p.argmax(1) if multiclass else (p >= .5).astype(int)
    average = 'macro' if multiclass else 'binary'
    metrics = dict(n=len(y),accuracy=accuracy_score(y,pred),
        precision=precision_score(y,pred,average=average,zero_division=0),
        recall=recall_score(y,pred,average=average,zero_division=0),
        f1=f1_score(y,pred,average=average,zero_division=0),
        roc_auc=np.nan,pr_auc=np.nan)
    if multiclass and len(np.unique(y)) == p.shape[1]:
        metrics['roc_auc'] = roc_auc_score(y,p,multi_class='ovr',labels=np.arange(p.shape[1]))
        areas = []
        for c in range(p.shape[1]):
            precision,recall,_ = precision_recall_curve(y==c,p[:,c])
            areas.append(auc(recall,precision))
        metrics['pr_auc'] = float(np.mean(areas))
    elif not multiclass and len(np.unique(y)) == 2:
        metrics['roc_auc'] = roc_auc_score(y,p)
        precision,recall,_ = precision_recall_curve(y,p)
        metrics['pr_auc'] = auc(recall,precision)
    if not multiclass:
        metrics['ece'] = compute_expected_calibration_error(p,y)[0]
        metrics['brier'] = compute_brier_score(p,y)
    return metrics


def regression(y, pred):
    return dict(mae=mean_absolute_error(y,pred),rmse=float(np.sqrt(mean_squared_error(y,pred))),r2=r2_score(y,pred))


def prediction_metrics(frame):
    return {**classification(frame.target_progression,frame.probability),
            **regression(frame.target_ctdna,frame.ctdna_forecast)}


def degradation(clean, perturbed, lower=False):
    absolute = perturbed-clean if lower else clean-perturbed
    return absolute, (100*absolute/abs(clean) if clean != 0 else np.nan)


def perturb_image(image, condition, seed, difficulty=.5):
    rng = np.random.default_rng(seed)
    if condition in ('brightness_minus','brightness_plus'):
        return ImageEnhance.Brightness(image).enhance(.75 if condition.endswith('minus') else 1.25)
    if condition in ('contrast_minus','contrast_plus'):
        return ImageEnhance.Contrast(image).enhance(.75 if condition.endswith('minus') else 1.25)
    if condition == 'blur':
        return image.filter(ImageFilter.GaussianBlur(1.5))
    if condition == 'compression':
        buffer = io.BytesIO()
        image.save(buffer,format='JPEG',quality=35)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')
    if condition in ('noise','stain','challenge'):
        a = np.asarray(image).astype(float)/255
        if condition in ('stain','challenge'):
            a *= rng.uniform(.8,1.2,(1,1,3))
        if condition in ('noise','challenge'):
            a += rng.normal(0,.05 if condition=='noise' else .025*difficulty,a.shape)
        image = Image.fromarray(np.uint8(np.clip(a,0,1)*255))
        if condition == 'challenge':
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(.85,1.15))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(.85,1.15))
        return image
    if condition == 'morphology':
        # Controlled spatial deformation proxy, not a claim of biological realism.
        return image.transform(image.size,Image.Transform.AFFINE,(1,.1,0,.06,1,0),resample=Image.Resampling.BILINEAR)
    if condition == 'ood_texture':
        return Image.fromarray(rng.integers(0,256,(*np.asarray(image).shape,),dtype=np.uint8))
    if condition != 'clean':
        raise ValueError(condition)
    return image


class ImageRows(Dataset):
    def __init__(self, frame, transform, condition='clean'):
        self.rows=frame.to_dict('records')
        self.transform,self.condition=transform,condition
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        path=dataset_v2.resolve_image_path(row['image_path'],row.get('relative_path',''))
        with Image.open(path) as image:
            image=image.convert('RGB')
            image=perturb_image(image,self.condition,stable_seed(row['tile_id']))
            return self.transform(image)


class Experiment:
    def __init__(self, smoke=False):
        self.smoke=smoke
        self.root=config.RESULTS_DIR/('smoke_v3' if smoke else 'validated_v3')
        self.root.mkdir(parents=True,exist_ok=True)
        self.checkpoints=config.CHECKPOINTS_DIR/('smoke_v3' if smoke else 'validated_v3')
        self.checkpoints.mkdir(parents=True,exist_ok=True)
        self.records=[]
        self.pathology={}
        self.temporal={}
        self.mil={}
        self.systems={}
        self.frames={}
        self.history={}
        self.meta=pd.read_csv(config.IMAGE_METADATA_PATH).sort_values(['patient_id','tile_id']).reset_index(drop=True)
        image_manifest=[]
        for row in self.meta.to_dict('records'):
            path=dataset_v2.resolve_image_path(row['image_path'],row.get('relative_path',''))
            image_manifest.append(dict(tile_id=row['tile_id'],sha256=sha(path)))
        write_json(self.root/'image_content_hashes.json',image_manifest)
        self.datasets={s:AuditedTemporalDataset(s) for s in SPLITS}
        self.epochs_image=int(os.environ.get('ONCOLOGY_PATHOLOGY_EPOCHS','15'))
        self.epochs_temporal=int(os.environ.get('ONCOLOGY_TEMPORAL_EPOCHS','20'))
        self.epochs_fusion=int(os.environ.get('ONCOLOGY_FUSION_EPOCHS','40'))
        if smoke:
            self.epochs_image=self.epochs_temporal=self.epochs_fusion=2
        self.data_fingerprint=hashlib.sha256((VERSION+sha(config.IMAGE_METADATA_PATH)+sha(config.BIOMARKERS_PATH)
            +sha(Path(__file__).parents[1]/'research_data.py')+sha(self.root/'image_content_hashes.json')).encode()).hexdigest()
        self.provenance=dict(version=VERSION,smoke=smoke,data_fingerprint=self.data_fingerprint,
            split_counts=audit_manifests(),seed=42,seeds=[42,123,2026],torch=torch.__version__,
            pathology_epochs=self.epochs_image,temporal_epochs=self.epochs_temporal,fusion_epochs=self.epochs_fusion,
            selection='Validation progression F1; ties: validation ROC-AUC, then lower ctDNA MAE; threshold 0.5 fixed.',
            forecast='Day90 landmark; nearest observed day120 ctDNA within +/-12 days; missing target excluded.',
            pathology_protocol='Frozen official pretrained encoder + independently optimized identical-width head; no end-to-end fine-tuning.',
            source_hashes={str(p.relative_to(config.STAGE_2_DIR)):sha(p) for p in (config.BASE_DIR).rglob('*.py')},
            exclusions={s:d.exclusions for s,d in self.datasets.items()})
        write_json(self.root/'protocol.json',self.provenance)
        self.verify_baselines()

    def verify_baselines(self):
        before=json.loads((config.DOCS_DIR/'baseline_integrity_before.json').read_text(encoding='utf-8-sig'))
        observed=[]
        for item in before:
            actual=sha(config.PROJECT_ROOT/item['path'])
            if actual.lower()!=item['sha256'].lower():
                raise ValueError(f"Protected baseline changed: {item['path']}")
            observed.append(dict(path=item['path'],sha256=actual,unchanged=True))
        write_json(self.root/'baseline_integrity_verified.json',observed)

    def record(self,name,split,metrics,domain,**extra):
        row=dict(model=name,split=split,domain=domain,**metrics,**extra)
        self.records.append(row)
        pd.DataFrame(self.records).to_csv(self.root/'model_comparison.csv',index=False)

    def frame(self,name,split,prob,ctdna):
        ds=self.datasets[split]
        data=pd.DataFrame(dict(patient_id=[i['patient_id'] for i in ds],
            target_progression=[float(i['target_progression']) for i in ds],
            target_ctdna=[float(i['target_ctdna']) for i in ds],
            target_day=[i['target_day'] for i in ds],
            last_input_day=[i['last_input_day'] for i in ds],
            probability=np.asarray(prob),ctdna_forecast=np.asarray(ctdna)))
        data['prediction']=(data.probability>=.5).astype(int)
        data.to_csv(self.root/f'predictions_{name}_{split}.csv',index=False)
        self.frames[name,split]=data
        return data

    def features(self,name,model,split,condition='clean'):
        frame=self.meta[self.meta.split.eq(split)].reset_index(drop=True)
        if self.smoke:
            frame=frame.groupby('class_label',group_keys=False).head(24).reset_index(drop=True)
        cache=self.root/f'embeddings_{name}_{split}_{condition}.pt'
        fingerprint=hashlib.sha256((self.data_fingerprint+model.weights_id+condition).encode()).hexdigest()
        if cache.exists():
            content=torch.load(cache,weights_only=False,map_location='cpu')
            if content['fingerprint']==fingerprint and content['tile_ids']==frame.tile_id.tolist():
                return content['features'],frame,content['seconds']
            raise ValueError(f'Stale cache requires a new run namespace: {cache}')
        model.eval()
        loader=DataLoader(ImageRows(frame,model.preprocess,condition),batch_size=32,shuffle=False,num_workers=0)
        features=[]
        start=time.perf_counter()
        with torch.inference_mode():
            for index,images in enumerate(loader):
                features.append(model.raw_features(images).cpu())
                if index%25==0:
                    print(f'  {name} {split} {condition}: {min((index+1)*32,len(frame))}/{len(frame)} tiles',flush=True)
        x=torch.cat(features)
        seconds=time.perf_counter()-start
        torch.save(dict(features=x,tile_ids=frame.tile_id.tolist(),fingerprint=fingerprint,seconds=seconds),cache)
        return x,frame,seconds

    def optimize(self,name,model,inputs,targets,val_inputs,val_targets,epochs,kind,seed=42,lr=1e-3):
        seed_all(seed)
        trainable=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=lr,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-6)
        best_score=-float('inf'); best=None; stale=0; logs=[]
        start=time.perf_counter()
        for epoch in range(epochs):
            model.train()
            total=0.
            for ids in torch.randperm(len(inputs[0])).split(32):
                optimizer.zero_grad()
                out=model(*(x[ids] for x in inputs))
                if kind in ('classifier','mil'):
                    logits=out[0] if kind=='mil' else out
                    loss=F.cross_entropy(logits,targets[0][ids].long())
                else:
                    c,l=(out['ctdna_forecast'],out['progression_logits']) if isinstance(out,dict) else out[:2]
                    loss=F.smooth_l1_loss(c,targets[0][ids])+F.binary_cross_entropy_with_logits(l,targets[1][ids])
                loss.backward()
                nn.utils.clip_grad_norm_(trainable,5.)
                optimizer.step()
                total+=float(loss.detach())*len(ids)
            scheduler.step()
            model.eval()
            with torch.no_grad():
                out=model(*val_inputs)
                if kind in ('classifier','mil'):
                    logits=out[0] if kind=='mil' else out
                    score=classification(val_targets[0].numpy(),logits.softmax(-1).numpy())['f1']
                    validation_loss=float(F.cross_entropy(logits,val_targets[0].long()))
                else:
                    c,l=(out['ctdna_forecast'],out['progression_logits']) if isinstance(out,dict) else out[:2]
                    # Predeclared combined validation objective for multi-task checkpoints.
                    validation_loss=float(F.smooth_l1_loss(c,val_targets[0])+F.binary_cross_entropy_with_logits(l,val_targets[1]))
                    score=-validation_loss
            logs.append(dict(epoch=epoch+1,training_loss=total/len(inputs[0]),validation_loss=validation_loss,selection_score=score))
            if score>best_score:
                best_score=score;best=copy.deepcopy(model.state_dict());stale=0
            else:
                stale+=1
            if (epoch+1)%5==0:
                print(f'  {name} seed {seed} epoch {epoch+1}: val loss={validation_loss:.5f}',flush=True)
            if stale>=6 and epoch>=9:
                break
        model.load_state_dict(best);model.eval()
        seconds=time.perf_counter()-start
        path=self.checkpoints/f'{name}_seed{seed}.pt'
        torch.save(dict(state_dict=model.state_dict(),seed=seed,version=VERSION,data_fingerprint=self.data_fingerprint,
            history=logs,training_seconds=seconds,best_score=best_score),path)
        pd.DataFrame(logs).to_csv(self.root/f'learning_curve_{name}_seed{seed}.csv',index=False)
        self.history[name,seed]=logs
        return seconds

    def train_pathology(self,name):
        print(f'PATHOLOGY: {name}',flush=True)
        seed_all(42)
        model=PretrainedPathology(name)
        raw={s:self.features(name,model,s) for s in SPLITS}
        # Same deterministic challenge view for every architecture; no test augmentation tuning.
        augmented=self.features(name,model,'train','challenge')
        train_x=torch.cat([raw['train'][0],augmented[0]])
        train_y=torch.tensor([config.CLASS_TO_IDX[x] for x in raw['train'][1].class_label]*2)
        val_y=torch.tensor([config.CLASS_TO_IDX[x] for x in raw['validation'][1].class_label])
        seconds=self.optimize(name,model.head,(train_x,),(train_y,),(raw['validation'][0],),(val_y,),self.epochs_image,'classifier',lr=5e-4)
        extraction=sum(v[2] for v in raw.values())+augmented[2]
        for split in ('validation','test'):
            x,metadata,infer_seconds=raw[split]
            with torch.no_grad():
                p=model.head(x).softmax(-1).numpy()
            y=np.array([config.CLASS_TO_IDX[x] for x in metadata.class_label])
            result=metadata[['patient_id','tile_id','class_label']].copy()
            for k in range(3):result[f'probability_{k}']=p[:,k]
            result.to_csv(self.root/f'tile_predictions_{name}_{split}.csv',index=False)
            self.record(name,split,classification(y,p),'tile histology',total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.head.parameters()),training_seconds=seconds,
                extraction_seconds=extraction,inference_ms_per_tile=1000*infer_seconds/len(metadata),weights=model.weights_id)
        self.pathology[name]=dict(model=model,raw=raw,training_seconds=seconds)
        torch.save(dict(state_dict=model.state_dict(),name=name,weights=model.weights_id,version=VERSION,data_fingerprint=self.data_fingerprint),self.checkpoints/f'pathology_{name}.pt')

    def temporal_tensors(self,split,challenge=None):
        ds=self.datasets[split] if challenge is None else AuditedTemporalDataset(split,challenge=challenge,norm_params=self.datasets['train'].norm_params)
        if [x['patient_id'] for x in ds] != [x['patient_id'] for x in self.datasets[split]]:
            raise ValueError('Challenge changed target cohort')
        return (torch.stack([x['features'] for x in ds]),torch.tensor([x['length'] for x in ds])),(torch.stack([x['target_ctdna'] for x in ds]),torch.stack([x['target_progression'] for x in ds]))

    def train_temporal(self,name,seed=42):
        print(f'TEMPORAL: {name} seed {seed}',flush=True)
        seed_all(seed)
        model=BiLSTMForecaster() if name=='bilstm' else ContinuousTemporalTransformer()
        tr,yt=self.temporal_tensors('train');va,yv=self.temporal_tensors('validation')
        seconds=self.optimize(name,model,tr,yt,va,yv,self.epochs_temporal,'temporal',seed)
        reps={}
        for split in SPLITS:
            x,y=self.temporal_tensors(split)
            start=time.perf_counter()
            with torch.no_grad():
                out=model(*x);c,l=out[:2]
                if name=='bilstm':
                    states,_=model.lstm(x[0]); rep=states[torch.arange(len(states)),x[1]-1]
                else:rep=out[2]
            reps[split]=rep.detach()
            elapsed=time.perf_counter()-start
            frame=self.frame(name if seed==42 else f'{name}_seed{seed}',split,l.sigmoid().numpy(),c.numpy())
            if split!='train':
                self.record(name,split,prediction_metrics(frame),'progression and ctDNA',seed=seed,
                    total_parameters=sum(p.numel() for p in model.parameters()),training_seconds=seconds,inference_ms_per_patient=elapsed*1000/len(frame))
        self.temporal[name]=dict(model=model,reps=reps)
        return model

    def bag_tensors(self,name,split,condition='clean'):
        model=self.pathology[name]['model']
        raw,metadata,_=(self.pathology[name]['raw'][split] if condition=='clean' else self.features(name,model,split,condition))
        lookup={tile:i for i,tile in enumerate(metadata.tile_id)}
        bags=composition_bags(metadata)
        x=torch.stack([torch.stack([raw[lookup[row['tile_id']]] for row in bag['rows']]) for bag in bags])
        y=torch.tensor([b['patient_label'] for b in bags])
        ids=[b['patient_id'] for b in bags]
        records=[dict(patient_id=b['patient_id'],position=k,source_tile_id=r['tile_id'],label=b['patient_label']) for b in bags for k,r in enumerate(b['rows'])]
        pd.DataFrame(records).to_csv(self.root/f'bag_manifest_{split}.csv',index=False)
        return x,y,ids

    def align(self, tensor, ids, split):
        lookup={p:i for i,p in enumerate(ids)}
        return tensor[torch.tensor([lookup[x['patient_id']] for x in self.datasets[split]])]

    def train_mil(self,mode,seed=42):
        print(f'MIL ResNet50: {mode}',flush=True)
        seed_all(seed)
        model=CachedMIL(2048,mode)
        data={s:self.bag_tensors('resnet50',s) for s in SPLITS}
        seconds=self.optimize(f'mil_{mode}',model,(data['train'][0],),(data['train'][1],),
            (data['validation'][0],),(data['validation'][1],),self.epochs_image,'mil',seed)
        reps={};probs={}
        for split,(x,y,ids) in data.items():
            with torch.no_grad():logit,rep,weights=model(x)
            p=logit.softmax(-1)
            reps[split]=self.align(rep,ids,split);probs[split]=self.align(p,ids,split)
            pd.DataFrame(weights.numpy(),index=ids).to_csv(self.root/f'tile_attention_{mode}_{split}.csv',index_label='patient_id')
            if split!='train':self.record(f'mil_{mode}',split,classification(y.numpy(),p.numpy()),'patient histology',training_seconds=seconds)
        self.mil[mode]=dict(model=model,reps=reps,probs=probs,data=data)

    def mean_pathology(self,name,split):
        x,_,ids=self.bag_tensors(name,split)
        model=self.pathology[name]['model']
        with torch.no_grad():
            probs=model.head(x.flatten(0,1)).softmax(-1).view(len(x),12,3).mean(1)
            reps=model.head[:2](x.flatten(0,1)).view(len(x),12,128).mean(1)
        return self.align(probs,ids,split),self.align(reps,ids,split)

    def fixed_system(self,name,path_name,temporal_name,attention=False):
        self.systems[name]=dict(kind='fixed',path_name=path_name,temporal_name=temporal_name,attention=attention)
        for split in SPLITS:
            p=self.mil['attention']['probs'][split] if attention else self.mean_pathology(path_name,split)[0]
            t=self.frames[temporal_name,split]
            out=FixedWeightedFusion()(p[:,1],torch.tensor(t.probability.to_numpy()),torch.tensor(t.ctdna_forecast.to_numpy()))
            frame=self.frame(name,split,out['multimodal_risk_score'].numpy(),t.ctdna_forecast.to_numpy())
            if split!='train':self.record(name,split,prediction_metrics(frame),'progression and ctDNA')

    def learned_system(self,name,fusion_type='gated',path_mode='attention',single=None,seed=42):
        seed_all(seed)
        p={s:self.mil[path_mode]['reps'][s] for s in SPLITS}
        t=self.temporal['transformer']['reps']
        if single=='pathology':t={s:torch.zeros_like(t[s]) for s in SPLITS}
        if single=='temporal':p={s:torch.zeros_like(p[s]) for s in SPLITS}
        model=build_fusion_model(fusion_type,p['train'].shape[1],t['train'].shape[1])
        inputs={s:(p[s],t[s]) for s in SPLITS}
        targets={s:self.temporal_tensors(s)[1] for s in SPLITS}
        seconds=self.optimize(name,model,inputs['train'],targets['train'],inputs['validation'],targets['validation'],self.epochs_fusion,'fusion',seed)
        for split in SPLITS:
            with torch.no_grad():out=model(*inputs[split])
            frame=self.frame(name,split,out['progression_prob'].numpy(),out['ctdna_forecast'].numpy())
            if split!='train':self.record(name,split,prediction_metrics(frame),'progression and ctDNA',seed=seed,training_seconds=seconds,total_parameters=sum(p.numel() for p in model.parameters()))
        self.systems[name]=dict(kind='learned',model=model,inputs=inputs,fusion_type=fusion_type,path_mode=path_mode,single=single)
        return model

    def select(self,candidates):
        def key(name):
            m=prediction_metrics(self.frames[name,'validation'])
            return m['f1'],m['roc_auc'] if np.isfinite(m['roc_auc']) else -1,-m['mae']
        return max(candidates,key=key)

    def ablations(self):
        mapping=dict(A='pathology_only',B='transformer',C='r50_attention_transformer_fixed',D=self.best_learned,
            E='r18_bilstm_fixed',F='r50_transformer_fixed',G='fusion_gated')
        rows=[]
        for id,name in mapping.items():
            rows.append(dict(ID=id,configuration=name,**prediction_metrics(self.frames[name,'test'])))
        self.ablation_rows=rows
        pd.DataFrame(rows).to_csv(self.root/'ablation.csv',index=False)

    def reliability(self):
        # Assess G explicitly so H is an actual uncertainty-evaluated G, not a copied row.
        name='fusion_gated'; system=self.systems[name];model=system['model']
        model.eval()
        with torch.no_grad():
            val=model(*system['inputs']['validation'])
            test=model(*system['inputs']['test'])
        scaler=TemperatureScaler()
        temperature=scaler.fit_validation_data(val['progression_logits'],self.temporal_tensors('validation')[1][1],max_iters=100,lr=.05)
        with torch.no_grad():cal=scaler(test['progression_logits']).sigmoid().numpy()
        mc=MCDropoutEvaluator(30).evaluate_classification(model,*system['inputs']['test'])
        raw=self.frames[name,'test']
        h=raw.copy();h['probability']=mc['mean_prediction'];h['prediction']=(h.probability>=.5).astype(int)
        h['uncertainty']=mc['uncertainty_score'];h['prediction_variance']=mc['predictive_variance']
        h.to_csv(self.root/'predictions_H_test.csv',index=False)
        self.h=h
        self.ablation_rows.append(dict(ID='H',configuration='G with MC mean prediction; OOD annotations added subsequently',**prediction_metrics(h)))
        pd.DataFrame(self.ablation_rows).to_csv(self.root/'ablation.csv',index=False)
        y=raw.target_progression.to_numpy()
        self.calibration=dict(model=name,temperature=temperature,raw_ece=compute_expected_calibration_error(raw.probability.to_numpy(),y)[0],
            calibrated_ece=compute_expected_calibration_error(cal,y)[0],raw_brier=compute_brier_score(raw.probability.to_numpy(),y),calibrated_brier=compute_brier_score(cal,y))
        raw['calibrated_probability']=cal
        raw.to_csv(self.root/'calibrated_predictions_test.csv',index=False)
        write_json(self.root/'calibration.json',self.calibration)
        torch.save(dict(temperature=scaler.state_dict(),model=name),self.checkpoints/'calibration.pt')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(5,5))
        for label,p in [('Raw',raw.probability.to_numpy()),('Temperature scaled',cal)]:
            _,bins=compute_expected_calibration_error(p,y)
            valid=np.array(bins['bin_counts'])>0
            ax.plot(np.array(bins['bin_confidences'])[valid],np.array(bins['bin_accuracies'])[valid],'o-',label=label)
        ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='Mean predicted probability',ylabel='Observed positive fraction',title='Test reliability: Gated fusion')
        ax.legend();fig.tight_layout();fig.savefig(self.root/'reliability_diagram.png',dpi=160);plt.close(fig)

    def ood(self):
        system=self.systems['fusion_gated'];model=system['model'];model.eval()
        with torch.no_grad():latent={s:model(*system['inputs'][s])['fused_representation'] for s in SPLITS}
        detector=LatentOODDetector();detector.fit_reference(latent['train']);detector.fit_threshold_on_validation(latent['validation'])
        clean=detector.predict(latent['test'])
        # Image-level unrelated noise textures passed through the full encoder/MIL/fusion path.
        x,_,ids=self.bag_tensors('resnet50','test','ood_texture')
        with torch.no_grad():
            _,p,_=self.mil['attention']['model'](x)
            p=self.align(p,ids,'test')
            shifted=model(p,system['inputs']['test'][1])['fused_representation']
        corrupt=detector.predict(shifted)
        labels=np.r_[np.zeros(len(latent['test'])),np.ones(len(shifted))]
        scores=np.r_[clean['mahalanobis_distances'],corrupt['mahalanobis_distances']]
        self.ood_results=dict(challenge='Independent RGB noise textures replacing pathology; temporal unchanged. Synthetic far-OOD only.',
            threshold=detector.threshold,clean_flag_rate=clean['shift_fraction'],shifted_flag_rate=corrupt['shift_fraction'],roc_auc=roc_auc_score(labels,scores))
        write_json(self.root/'ood.json',self.ood_results)
        self.h['ood_score']=clean['mahalanobis_distances'];self.h['ood']=clean['ood_flags']
        self.h.to_csv(self.root/'predictions_H_test.csv',index=False)
        torch.save(dict(centroid=detector.centroid,inv_cov=detector.inv_cov,threshold=detector.threshold),self.checkpoints/'ood.pt')
        return detector

    def robustness(self):
        records=[]
        conditions=['brightness_minus','brightness_plus','contrast_minus','contrast_plus','stain','blur','noise','compression','morphology']
        # Both baseline and upgrade use the identical patient-composition task and perturbations.
        for name in ['resnet18','resnet50']:
            model=self.pathology[name]['model'];raw,meta,_=self.pathology[name]['raw']['test']
            y=np.array([config.CLASS_TO_IDX[c] for c in meta.class_label])
            with torch.no_grad():clean=classification(y,model.head(raw).softmax(-1).numpy())['f1']
            for condition in conditions:
                x,_,_=self.features(name,model,'test',condition)
                with torch.no_grad():value=classification(y,model.head(x).softmax(-1).numpy())['f1']
                absolute,percent=degradation(clean,value)
                records.append(dict(model=name,condition=condition,metric='tile_macro_f1',clean=clean,perturbed=value,absolute_degradation=absolute,percentage_degradation=percent))
        for name in ['bilstm','transformer']:
            model=self.temporal[name]['model'];model.eval();clean=prediction_metrics(self.frames[name,'test'])
            for condition in ['missing_visits','short_histories','irregular_intervals','measurement_noise','temporary_spikes','outliers','missing_biomarkers']:
                x,_=self.temporal_tensors('test',condition)
                with torch.no_grad():c,l=model(*x)[:2]
                frame=self.frames[name,'test'].copy();frame['probability']=l.sigmoid().numpy();frame['ctdna_forecast']=c.numpy()
                frame.to_csv(self.root/f'robust_predictions_{name}_{condition}.csv',index=False)
                metrics=prediction_metrics(frame)
                for metric in ['f1','mae']:
                    absolute,percent=degradation(clean[metric],metrics[metric],metric=='mae')
                    records.append(dict(model=name,condition=condition,metric=metric,clean=clean[metric],perturbed=metrics[metric],absolute_degradation=absolute,percentage_degradation=percent))
        system=self.systems[self.best_learned];model=system['model'];p,t=system['inputs']['test']
        clean=prediction_metrics(self.frames[self.best_learned,'test'])
        for mode,x in [('FULL_MULTIMODAL',(p,t)),('PATHOLOGY_ONLY',(p,torch.zeros_like(t))),('TEMPORAL_ONLY',(torch.zeros_like(p),t))]:
            with torch.no_grad():out=model(*x)
            frame=self.frames[self.best_learned,'test'].copy();frame['probability']=out['progression_prob'].numpy();frame['ctdna_forecast']=out['ctdna_forecast'].numpy()
            frame.to_csv(self.root/f'robust_predictions_{mode}.csv',index=False)
            metrics=prediction_metrics(frame)
            for metric in ['f1','mae']:
                absolute,percent=degradation(clean[metric],metrics[metric],metric=='mae')
                records.append(dict(model=self.best_learned,condition=mode,metric=metric,clean=clean[metric],perturbed=metrics[metric],absolute_degradation=absolute,percentage_degradation=percent))
        records.append(dict(model=self.best_learned,condition='INSUFFICIENT_DATA',metric='prediction_coverage',clean=1.,perturbed=0.,absolute_degradation=1.,percentage_degradation=100.))
        self.robustness_results=pd.DataFrame(records)
        self.robustness_results.to_csv(self.root/'robustness.csv',index=False)

    def multiseed(self):
        rows=[]
        chosen=self.systems[self.best_learned]
        for seed in [42,123,2026]:
            name=f'final_fusion_seed{seed}'
            self.learned_system(name,chosen['fusion_type'],chosen['path_mode'],seed=seed)
            rows.append(dict(seed=seed,architecture=self.best_learned,scope='Independent fusion retraining conditional on fixed seed42 encoders',**prediction_metrics(self.frames[name,'test'])))
        self.seed_results=pd.DataFrame(rows)
        self.seed_results.to_csv(self.root/'multi_seed_evaluation.csv',index=False)
        # Required public path has raw seed rows, not strings mixed with numeric fields.
        self.seed_results.to_csv(config.RESULTS_DIR/'multi_seed_evaluation.csv',index=False)
        write_json(self.root/'multi_seed_summary.json',{m:dict(mean=float(self.seed_results[m].mean()),sd=float(self.seed_results[m].std(ddof=1))) for m in ['f1','roc_auc','mae','r2']})

    def failure_analysis(self):
        frame=self.h.copy()
        frame['failure']=np.select([(frame.target_progression==0)&(frame.probability>=.5),(frame.target_progression==1)&(frame.probability<.5)],['false_positive','false_negative'],default='correct')
        frame['ctdna_absolute_error']=(frame.ctdna_forecast-frame.target_ctdna).abs()
        frame.to_csv(self.root/'failure_analysis.csv',index=False)
        self.failures=frame
        # All attribution methods operate on actual trained Transformer predictions.
        model=self.temporal['transformer']['model'];x,y=self.temporal_tensors('test');model.eval()
        clean=prediction_metrics(self.frames['transformer','test'])['f1']
        rows=[]
        for k,feature in enumerate(config.ALL_TEMPORAL_FEATURES):
            for method in ['permutation','feature_ablation']:
                pert=x[0].clone()
                if method=='permutation':
                    generator=torch.Generator().manual_seed(42+k)
                    for visit in range(pert.shape[1]):
                        valid=torch.where(x[1]>visit)[0]
                        permutation=valid[torch.randperm(len(valid),generator=generator)]
                        pert[valid,visit,k]=x[0][permutation,visit,k]
                else:pert[:,:,k]=0
                with torch.no_grad():_,logits,_=model(pert,x[1])
                score=classification(y[1].numpy(),logits.sigmoid().numpy())['f1']
                rows.append(dict(method=method,feature=feature,clean_f1=clean,perturbed_f1=score,signed_f1_drop=clean-score))
        for visit in range(x[0].shape[1]):
            pert=x[0].clone();pert[:,visit,:]=0
            with torch.no_grad():_,logits,_=model(pert,x[1])
            score=classification(y[1].numpy(),logits.sigmoid().numpy())['f1']
            rows.append(dict(method='visit_occlusion',feature=f'visit_index_{visit}',clean_f1=clean,perturbed_f1=score,signed_f1_drop=clean-score))
        pd.DataFrame(rows).to_csv(self.root/'temporal_attributions.csv',index=False)
        # ResNet50 Grad-CAM with input gradients; saved original image and heatmap.
        model=self.pathology['resnet50']['model'];layer=model.encoder.layer4[-1];activations=[];gradients=[]
        def hook(module,inputs,out):
            activations.append(out)
            out.register_hook(lambda grad:gradients.append(grad))
        handle=layer.register_forward_hook(hook)
        row=self.meta[self.meta.split.eq('test')].iloc[0]
        path=dataset_v2.resolve_image_path(row.image_path,row.get('relative_path',''))
        with Image.open(path) as im:original=im.convert('RGB');image=model.preprocess(original).unsqueeze(0).requires_grad_(True)
        model.zero_grad();logits=model(image);class_id=int(logits.argmax(1));logits[0,class_id].backward();handle.remove()
        cam=(gradients[0].mean((2,3),keepdim=True)*activations[0]).sum(1,keepdim=True).relu()
        cam=F.interpolate(cam,(224,224),mode='bilinear',align_corners=False)[0,0].detach().numpy()
        cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(7,3));axes[0].imshow(original);axes[1].imshow(cam,cmap='magma')
        for axis in axes:axis.axis('off')
        axes[0].set_title(str(row.tile_id));axes[1].set_title(f'ResNet50 class {class_id} Grad-CAM')
        fig.tight_layout();fig.savefig(self.root/'resnet50_gradcam.png',dpi=150);plt.close(fig)

    def reports(self):
        def table(frame):
            cols=frame.columns.tolist()
            lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
            for _,row in frame.iterrows():
                cells=[]
                for value in row:
                    cells.append('undefined' if pd.isna(value) else f'{value:.4f}' if isinstance(value,(float,np.floating)) else str(value))
                lines.append('| '+' | '.join(cells)+' |')
            return '\n'.join(lines)
        comparison=pd.DataFrame(self.records)
        test=comparison[comparison.split.eq('test')]
        columns=['model','domain','f1','roc_auc','pr_auc','mae','rmse','r2','ece','brier']
        model_md='# Model comparison\n\n'+table(test.reindex(columns=columns))
        model_md+='\n\nHistology and progression are different targets; their F1 values must not be compared directly. Timings, counts, complete metrics and weights are in model_comparison.csv.\n'
        robust_md='# Robustness report\n\n'+table(self.robustness_results)+'\n\nNegative degradation is an observed improvement. Percentage degradation is undefined when the clean metric is zero. Missing-modality challenges are zero-embedding stress tests of the full model; insufficient data abstains.\n'
        cal_md='# Calibration report\n\n'+table(pd.DataFrame([self.calibration]))+'\n\nTemperature was fitted on validation only. Test reliability diagram: `../results/validated_v3/reliability_diagram.png`. MC uncertainty is prediction spread, not a clinical confidence interval. Calibration degradation is reported without clipping.\n'
        failure_md='# Failure analysis\n\n'+table(self.failures.failure.value_counts().rename_axis('type').reset_index(name='count'))
        failure_md+='\n\nActual largest ctDNA errors:\n\n'+table(self.failures.nlargest(10,'ctdna_absolute_error')[['patient_id','failure','target_ctdna','ctdna_forecast','ctdna_absolute_error','uncertainty','ood']])
        failure_md+='\n\nAll patient predictions, false positives/negatives, uncertainty and OOD flags are in failure_analysis.csv. Temporal and missing-modality challenge predictions are saved separately. Attributions are experimental sensitivities, not biological explanations.\n'
        def f1(name,domain='tile histology'):
            return float(test[(test.model==name)&(test.domain==domain)].iloc[0].f1)
        answers=[
            f"ResNet50 vs ResNet18 tile F1: {f1('resnet50'):.4f} vs {f1('resnet18'):.4f}. Observed difference {f1('resnet50')-f1('resnet18'):+.4f}; one training seed, no claim of statistical superiority.",
            f"Attention vs mean patient-histology F1: {f1('mil_attention','patient histology'):.4f} vs {f1('mil_mean','patient histology'):.4f}.",
            f"Transformer vs BiLSTM progression F1: {prediction_metrics(self.frames['transformer','test'])['f1']:.4f} vs {prediction_metrics(self.frames['bilstm','test'])['f1']:.4f}.",
            f"Validation-selected learned fusion: {self.best_learned}. Fixed/learned comparisons use actual common-cohort progression predictions below.",
            f"Overall validation-selected system: {self.recommended}. Single-modality candidates were included; see validation rows in CSV.",
            'Three seeds independently retrain the learned fusion head. Stability is conditional on fixed feature encoders; it is not full-pipeline multi-seed stability.',
            f"Gated fusion test ECE: {self.calibration['raw_ece']:.4f} raw, {self.calibration['calibrated_ece']:.4f} calibrated.",
            'Perturbation results, including any improvements, are reported in the robustness table.',
            'Missing modalities are explicitly tested; zero-embedding degradation is not evidence of a trained fallback. No-input requests must abstain.',
            f"Far-OOD synthetic texture ROC-AUC: {self.ood_results['roc_auc']:.4f}; clean flag rate {self.ood_results['clean_flag_rate']:.4f}. Real OOD sensitivity remains unvalidated.",
            f"False positives: {(self.failures.failure=='false_positive').sum()}; false negatives: {(self.failures.failure=='false_negative').sum()}. See actual patient-level failure table.",
            f"Recommended research configuration by validation: {self.recommended}. The expected ResNet50/attention/Transformer/gated system was not forced to win."]
        final='# Stage 2 DL experimental report\n\n'+ '\n\n'.join(f'{i+1}. {x}' for i,x in enumerate(answers))
        final+='\n\n## Actual ablations\n\n'+table(pd.DataFrame(self.ablation_rows))
        final+='\n\n## Independent fusion seeds\n\n'+table(self.seed_results[['seed','f1','roc_auc','mae','r2']])
        summary=pd.DataFrame([dict(metric=m,mean=self.seed_results[m].mean(),sd=self.seed_results[m].std(ddof=1)) for m in ['f1','roc_auc','mae','r2']])
        final+='\n\n'+table(summary)
        final+='\n\n## Limitations\n\nSynthetic data only; research-only, no clinical validation. The bag-composition challenge resamples each patient’s own synthetic tiles and does not create independent tissue. Pathology was generated independently of temporal prognosis; no outcome-conditioned pathology shortcut was added. The exact preserved BiLSTM has padding-sensitive backward states. Frozen pretrained backbones were not fine-tuned end-to-end. Missing target patients are excluded using a predeclared observation window; exclusions are in protocol.json. Classification is a synthetic future biomarker trend, not diagnosed recurrence. Validation is reused for selection and calibration, which may increase selection uncertainty. No clinical OOD data are available. See protocol.json and prediction CSVs for reproducibility.\n'
        for file,body in [('MODEL_COMPARISON.md',model_md),('ROBUSTNESS_REPORT.md',robust_md),('CALIBRATION_REPORT.md',cal_md),('FAILURE_ANALYSIS.md',failure_md),('STAGE_2_DL_FINAL_REPORT.md',final)]:
            (config.DOCS_DIR/file).write_text(body,encoding='utf-8')
        write_json(self.root/'selection.json',dict(recommended=self.recommended,best_learned=self.best_learned,selection_partition='validation'))


def main(smoke_test=False):
    if smoke_test:
        raise ValueError('Smoke tests use pytest; final experiment runner refuses incomplete cohorts.')
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    torch.hub.set_dir(str(config.CHECKPOINTS_DIR/'pretrained'))
    seed_all(42)
    e=Experiment(False)
    for name in ['resnet18','resnet50','efficientnet_b0','vit']:e.train_pathology(name)
    e.train_temporal('bilstm');e.train_temporal('transformer')
    for mode in ['mean','median','max','attention']:e.train_mil(mode)
    e.fixed_system('r18_bilstm_fixed','resnet18','bilstm')
    e.fixed_system('r50_bilstm_fixed','resnet50','bilstm',True)
    e.fixed_system('r50_transformer_fixed','resnet50','transformer')
    e.fixed_system('r50_attention_transformer_fixed','resnet50','transformer',True)
    for fusion in ['concat_mlp','gated','cross_attention']:e.learned_system(f'fusion_{fusion}',fusion)
    e.learned_system('pathology_only',single='pathology')
    e.best_learned=e.select(['fusion_concat_mlp','fusion_gated','fusion_cross_attention'])
    e.recommended=e.select(['r18_bilstm_fixed','r50_bilstm_fixed','r50_transformer_fixed','r50_attention_transformer_fixed',e.best_learned,'pathology_only','bilstm','transformer'])
    write_json(e.root/'selection_locked_before_reliability.json',dict(recommended=e.recommended,best_learned=e.best_learned))
    e.ablations();e.reliability();e.ood();e.robustness();e.multiseed();e.failure_analysis();e.reports();e.verify_baselines()
    print(f'Completed: {e.root}; validation recommendation={e.recommended}',flush=True)


if __name__=='__main__':
    main()
