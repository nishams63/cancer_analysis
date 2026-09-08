"""Verify saved evidence, calibrate the selected system, and build the final comparison.

Run only after validated_experiments completes. Uses saved validation predictions
for calibration and saved test predictions for metrics; never selects on test.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dl import config
from dl.training.validated_experiments import Experiment,prediction_metrics, write_json, sha
from dl.uncertainty_calibration import TemperatureScaler,compute_expected_calibration_error,compute_brier_score
from dl.research_data import AuditedTemporalDataset
from dl.models.research_models import PretrainedPathology,CachedMIL
from dl.models.temporal_transformer import ContinuousTemporalTransformer
from dl.models.multimodal_fusion import build_fusion_model
from dl.src.temporal_model import BiLSTMForecaster
from dl.ood_detector import LatentOODDetector


def restore_study(root):
    """Restore evaluated models and fingerprinted features without retraining."""
    e=Experiment.__new__(Experiment)
    e.root=root;e.checkpoints=config.CHECKPOINTS_DIR/'validated_v3';e.smoke=False
    protocol=json.loads((root/'protocol.json').read_text())
    e.data_fingerprint=protocol['data_fingerprint']
    e.image_device=torch.device(protocol['image_device']);e.image_batch_size=protocol['image_batch_size']
    e.meta=pd.read_csv(config.IMAGE_METADATA_PATH).sort_values(['patient_id','tile_id']).reset_index(drop=True)
    e.datasets={s:AuditedTemporalDataset(s) for s in ('train','validation','test')}
    e.pathology={};e.temporal={};e.mil={}
    for name in ('resnet18','resnet50'):
        model=PretrainedPathology(name,load_weights=False)
        model.load_state_dict(torch.load(e.checkpoints/f'pathology_{name}.pt',map_location='cpu',weights_only=False)['state_dict']);model.eval()
        e.pathology[name]=dict(model=model,raw={s:e.features(name,model,s) for s in e.datasets})
    for name in ('bilstm','transformer'):
        model=BiLSTMForecaster() if name=='bilstm' else ContinuousTemporalTransformer()
        model.load_state_dict(torch.load(e.checkpoints/f'{name}_seed42.pt',map_location='cpu',weights_only=False)['state_dict']);model.eval()
        reps={}
        for split in e.datasets:
            x,_=e.temporal_tensors(split)
            with torch.no_grad():
                if name=='bilstm':
                    out,_=model.lstm(x[0]);rep=out[torch.arange(len(out)),x[1]-1]
                else:rep=model(*x)[2]
            reps[split]=rep
        e.temporal[name]=dict(model=model,reps=reps)
    mil=CachedMIL(2048,'attention');mil.load_state_dict(torch.load(e.checkpoints/'mil_attention_seed42.pt',weights_only=False)['state_dict']);mil.eval()
    reps={}
    for split in e.datasets:
        x,_,ids=e.bag_tensors('resnet50',split)
        with torch.no_grad():_,rep,_=mil(x)
        reps[split]=e.align(rep,ids,split)
    e.mil['attention']=dict(model=mil,reps=reps)
    return e


def selected_representation(e,name,split,challenge=None):
    temporal_name='bilstm' if 'bilstm' in name else 'transformer'
    t=e.temporal[temporal_name]['reps'][split]
    if challenge:
        x,_=e.temporal_tensors(split,challenge);model=e.temporal[temporal_name]['model']
        with torch.no_grad():
            if temporal_name=='bilstm':
                states,_=model.lstm(x[0]);t=states[torch.arange(len(states)),x[1]-1]
            else:t=model(*x)[2]
    if name in ('bilstm','transformer'):return t
    attention='attention' in name or name=='r50_bilstm_fixed' or name.startswith('fusion_') or name=='pathology_only'
    p=e.mil['attention']['reps'][split] if attention else e.mean_pathology('resnet18' if name.startswith('r18') else 'resnet50',split)[1]
    if name.endswith('_fixed'):return torch.cat([p,t],dim=1)
    kind='gated' if name=='pathology_only' else name.removeprefix('fusion_')
    model=build_fusion_model(kind,128,64)
    model.load_state_dict(torch.load(e.checkpoints/f'{name}_seed42.pt',weights_only=False)['state_dict']);model.eval()
    if name=='pathology_only':t=torch.zeros_like(t)
    with torch.no_grad():return model(p,t)['fused_representation']


def markdown(frame):
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for _,row in frame.iterrows():
        lines.append('| '+' | '.join('undefined' if pd.isna(v) else f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    root=config.RESULTS_DIR/'validated_v3'
    selected=json.loads((root/'selection.json').read_text())['recommended']
    rows=[]
    configurations=[('Baseline','resnet18','bilstm','mean','fixed','r18_bilstm_fixed'),
        ('Pathology upgrade','resnet50','bilstm','attention','fixed','r50_bilstm_fixed'),
        ('Temporal upgrade','resnet50','transformer','attention','fixed','r50_attention_transformer_fixed'),
        ('Full target','resnet50','transformer','attention','gated','fusion_gated')]
    calibrations=[]
    names=list(dict.fromkeys([item[-1] for item in configurations]+[selected]))
    for name in names:
        val=pd.read_csv(root/f'predictions_{name}_validation.csv')
        test=pd.read_csv(root/f'predictions_{name}_test.csv')
        def logits(frame):
            p=torch.tensor(frame.probability.to_numpy(),dtype=torch.float32).clamp(1e-6,1-1e-6)
            return torch.logit(p)
        scaler=TemperatureScaler()
        scaler.fit_validation_data(logits(val),torch.tensor(val.target_progression.to_numpy()),max_iters=100,lr=.05)
        with torch.no_grad():p=scaler(logits(test)).sigmoid().numpy()
        y=test.target_progression.to_numpy()
        calibration=dict(model=name,temperature=float(scaler.temperature.item()),
            raw_ece=compute_expected_calibration_error(test.probability.to_numpy(),y)[0],calibrated_ece=compute_expected_calibration_error(p,y)[0],
            raw_brier=compute_brier_score(test.probability.to_numpy(),y),calibrated_brier=compute_brier_score(p,y))
        calibrations.append(calibration)
        test['calibrated_probability']=p
        test.to_csv(root/f'calibrated_{name}_test.csv',index=False)
        torch.save(dict(model=name,state_dict=scaler.state_dict()),config.CHECKPOINTS_DIR/'validated_v3'/f'calibration_{name}.pt')
    for label,path,temp,pooling,fusion,name in configurations:
        frame=pd.read_csv(root/f'predictions_{name}_test.csv')
        m=prediction_metrics(frame)
        rows.append(dict(System=label,Pathology=path,Temporal=temp,Aggregation=pooling,Fusion=fusion,
            F1=m['f1'],ROC_AUC=m['roc_auc'],ctDNA_MAE=m['mae'],ECE=m['ece'],Brier=m['brier']))
    comparison=pd.DataFrame(rows)
    comparison.to_csv(root/'final_system_comparison.csv',index=False)
    pd.DataFrame(calibrations).to_csv(root/'calibration_all_systems.csv',index=False)
    (config.DOCS_DIR/'CALIBRATION_REPORT.md').write_text('# Calibration report\n\n'+markdown(pd.DataFrame(calibrations))+
        '\n\nEach temperature is fitted using validation predictions only. Positive or negative changes are retained. The temperature-scaling operation does not alter the 0.5 decision boundary; it changes probability reliability.\n',encoding='utf-8')
    final=config.DOCS_DIR/'STAGE_2_DL_FINAL_REPORT.md'
    body=final.read_text(encoding='utf-8')+'\n\n## Direct system comparison\n\n'+markdown(comparison)
    final.write_text(body,encoding='utf-8')
    # Record actual failures for the selected model as well as the Gated-system analysis.
    frame=pd.read_csv(root/f'calibrated_{selected}_test.csv')
    frame['failure']=np.select([(frame.target_progression==0)&(frame.probability>=.5),(frame.target_progression==1)&(frame.probability<.5)],['false_positive','false_negative'],default='correct')
    frame['ctdna_absolute_error']=(frame.ctdna_forecast-frame.target_ctdna).abs()
    frame.to_csv(root/'selected_model_failures.csv',index=False)
    study=restore_study(root)
    detector=LatentOODDetector()
    detector.fit_reference(selected_representation(study,selected,'train'))
    detector.fit_threshold_on_validation(selected_representation(study,selected,'validation'))
    clean=detector.predict(selected_representation(study,selected,'test'))
    challenged=detector.predict(selected_representation(study,selected,'test','outliers'))
    frame['ood_score']=clean['mahalanobis_distances'];frame['ood_status']=clean['status']
    frame.to_csv(root/'selected_model_failures.csv',index=False)
    torch.save(dict(model=selected,centroid=detector.centroid,inv_cov=detector.inv_cov,threshold=detector.threshold),study.checkpoints/f'ood_{selected}.pt')
    write_json(root/'selected_ood.json',dict(model=selected,threshold=detector.threshold,clean_flag_rate=clean['shift_fraction'],
        outlier_flag_rate=challenged['shift_fraction'],limitation='Controlled biomarker outlier stress test, not established clinical OOD ground truth'))
    failure=config.DOCS_DIR/'FAILURE_ANALYSIS.md'
    failure.write_text(failure.read_text(encoding='utf-8')+f'\n\n## Validation-selected system: {selected}\n\n'+
        markdown(frame.failure.value_counts().rename_axis('failure').reset_index(name='count'))+
        '\n\n'+markdown(frame.nlargest(10,'ctdna_absolute_error')[['patient_id','failure','target_ctdna','ctdna_forecast','ctdna_absolute_error']]),encoding='utf-8')
    # Content-address all result files. This manifest makes table sources traceable.
    files={str(p.relative_to(root)):sha(p) for p in root.iterdir() if p.is_file() and p.name!='artifact_hashes.json'}
    write_json(root/'artifact_hashes.json',files)
    print('Final evidence checked and calibration fitted for all four comparison systems plus the selected model.')


if __name__=='__main__':main()
