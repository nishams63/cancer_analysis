"""Artifact-backed inference for the validation-selected research system.

Reliability outputs are available only when fitted artifacts match the model.
No checkpoint or absent modality is replaced by random predictions.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from . import config
from .research_data import AuditedTemporalDataset, prepare_history
from .models.research_models import PretrainedPathology, CachedMIL
from .models.temporal_transformer import ContinuousTemporalTransformer
from .models.multimodal_fusion import build_fusion_model, FixedWeightedFusion
from .src.temporal_model import BiLSTMForecaster
from .uncertainty_calibration import MCDropoutEvaluator, TemperatureScaler
from .ood_detector import LatentOODDetector


def modality_status(pathology,temporal):
    return ('FULL_MULTIMODAL' if pathology and temporal else 'PATHOLOGY_ONLY' if pathology
            else 'TEMPORAL_ONLY' if temporal else 'INSUFFICIENT_DATA')


class ResearchPredictor:
    _instance=None
    @classmethod
    def ready(cls):
        return (config.RESULTS_DIR/'validated_v3/selection.json').exists()
    @classmethod
    def shared(cls):
        if cls._instance is None:cls._instance=cls()
        return cls._instance

    def __init__(self):
        if not self.ready():raise RuntimeError('Validated experiments are not complete; no recommended model is available')
        self.results=config.RESULTS_DIR/'validated_v3'
        self.checkpoints=config.CHECKPOINTS_DIR/'validated_v3'
        selection=json.loads((self.results/'selection.json').read_text())
        self.selected=selection['recommended'];self.best_learned=selection['best_learned']
        self.norm=AuditedTemporalDataset('train').norm_params
        self.models={}

    def load_temporal(self,name):
        if name not in self.models:
            model=BiLSTMForecaster() if name=='bilstm' else ContinuousTemporalTransformer()
            checkpoint=torch.load(self.checkpoints/f'{name}_seed42.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(checkpoint['state_dict']);model.eval();self.models[name]=model
        return self.models[name]

    def load_pathology(self,name):
        if name not in self.models:
            model=PretrainedPathology(name,load_weights=False)
            checkpoint=torch.load(self.checkpoints/f'pathology_{name}.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(checkpoint['state_dict']);model.eval();self.models[name]=model
        return self.models[name]

    def load_mil(self):
        if 'mil' not in self.models:
            model=CachedMIL(2048,'attention')
            model.load_state_dict(torch.load(self.checkpoints/'mil_attention_seed42.pt',map_location='cpu',weights_only=False)['state_dict'])
            model.eval();self.models['mil']=model
        return self.models['mil']

    def load_fusion(self,name):
        if name not in self.models:
            kind='gated' if name=='pathology_only' else name.removeprefix('fusion_')
            model=build_fusion_model(kind,128,64)
            model.load_state_dict(torch.load(self.checkpoints/f'{name}_seed42.pt',map_location='cpu',weights_only=False)['state_dict'])
            model.eval();self.models[name]=model
        return self.models[name]

    def predict(self,patient_id,tile_paths=None,observations=None):
        has_p=bool(tile_paths);has_t=bool(observations)
        status=modality_status(has_p,has_t)
        result=dict(patient_id=patient_id,modality_status=status,available_modalities=[x for x,b in [('pathology',has_p),('temporal',has_t)] if b],
            progression_probability=None,ctdna_forecast_30d=None,risk_level=None,confidence=None,uncertainty=None,
            prediction_variance=None,ood_score=None,ood_status='NOT_EVALUATED',model_configuration=None,
            calibration_status='NOT_CALIBRATED',mandatory_disclaimer=config.MANDATORY_DISCLAIMER)
        if status=='INSUFFICIENT_DATA':return result
        selected=self.selected
        # Missing-modality fallback uses a genuinely trained single-modality model.
        if not has_p:selected='transformer'
        if not has_t:selected='pathology_only'
        result['model_configuration']=selected
        temporal_name='bilstm' if 'bilstm' in selected else 'transformer'
        c=prob=uncertainty=variance=None
        t=torch.zeros(1,64)
        model=self.load_temporal(temporal_name) if has_t else None
        if has_t:
            history=pd.DataFrame(observations)
            for lab in config.TEMPORAL_NUMERICAL_FEATURES[:5]:
                if lab not in history:history[lab]=np.nan
            features,length,_=prepare_history(history,self.norm)
            inputs=(features.unsqueeze(0),torch.tensor([length]))
            with torch.no_grad():
                out=model(*inputs);c,logit=out[:2]
                if temporal_name=='bilstm':
                    states,_=model.lstm(inputs[0]);t=states[:,length-1]
                else:t=out[2]
                prob=logit.sigmoid()
            if selected in ('bilstm','transformer'):
                mc=MCDropoutEvaluator(30).evaluate_classification(model,*inputs)
                uncertainty=float(mc['uncertainty_score'][0]);variance=float(mc['predictive_variance'][0])
        if selected not in ('bilstm','transformer'):
            pathology_name='resnet18' if selected.startswith('r18') else 'resnet50'
            pathology=self.load_pathology(pathology_name)
            images=[]
            for path in tile_paths:
                with Image.open(path) as image:images.append(pathology.preprocess(image.convert('RGB')))
            with torch.no_grad():
                raw=pathology.raw_features(torch.stack(images))
                if 'attention' in selected or selected.startswith('fusion_') or selected=='pathology_only':
                    plogit,p,weights=self.load_mil()(raw.unsqueeze(0))
                    malignant=plogit.softmax(-1)[:,1]
                    result['tile_attention']=[dict(tile_path=str(tile_paths[i]),weight=float(weights[0,i])) for i in weights[0].argsort(descending=True)]
                else:
                    malignant=pathology.head(raw).softmax(-1)[:,1].mean().reshape(1)
                    p=pathology.head[:2](raw).mean(0,keepdim=True)
            if selected.endswith('_fixed'):
                prob=FixedWeightedFusion()(malignant,prob,c)['multimodal_risk_score']
            else:
                fusion=self.load_fusion(selected)
                with torch.no_grad():out=fusion(p,t);prob=out['progression_prob'];c=out['ctdna_forecast']
                mc=MCDropoutEvaluator(30).evaluate_classification(fusion,p,t)
                uncertainty=float(mc['uncertainty_score'][0]);variance=float(mc['predictive_variance'][0])
                if selected=='fusion_gated' and status=='FULL_MULTIMODAL':
                    scaler=TemperatureScaler();scaler.load_state_dict(torch.load(self.checkpoints/'calibration.pt',weights_only=False)['temperature'])
                    with torch.no_grad():prob=scaler(out['progression_logits']).sigmoid()
                    result['calibration_status']='VALIDATION_TEMPERATURE_SCALING'
                    artifact=torch.load(self.checkpoints/'ood.pt',weights_only=False)
                    detector=LatentOODDetector();detector.centroid=artifact['centroid'];detector.inv_cov=artifact['inv_cov'];detector.threshold=artifact['threshold'];detector.fitted=True
                    ood=detector.predict(out['fused_representation'])
                    result['ood_score']=float(ood['mahalanobis_distances'][0]);result['ood_status']=ood['status'][0]
        probability=float(prob[0]);forecast=float(c[0])
        if not np.isfinite([probability,forecast]).all():raise RuntimeError('Nonfinite model output')
        result.update(progression_probability=probability,ctdna_forecast_30d=forecast,
            risk_level='HIGH' if probability>=.65 else 'MODERATE' if probability>=.35 else 'LOW',
            confidence=max(probability,1-probability),uncertainty=uncertainty,prediction_variance=variance,
            confidence_definition='Maximum predicted class probability; not a guarantee of correctness',
            risk_confidence_status=('HIGH_RISK_LOW_CONFIDENCE' if probability>=.65 and uncertainty is not None and uncertainty>=.2
                else 'HIGH_RISK_HIGH_CONFIDENCE' if probability>=.8 and uncertainty is not None and uncertainty<.2 else 'INTERMEDIATE'))
        return result
