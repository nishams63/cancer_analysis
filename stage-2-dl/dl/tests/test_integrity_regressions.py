"""Regression tests for the scientific-integrity defects found in the initial audit."""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dl import config
from dl.research_data import AuditedTemporalDataset, audit_manifests, composition_bags, prepare_history
from dl.models import pathology_benchmarks as pb
from dl.models.research_models import CachedMIL
from dl.models.temporal_transformer import ContinuousTemporalTransformer
from dl.training.validated_experiments import classification, degradation
from dl.uncertainty_calibration import MCDropoutEvaluator


def test_manifest_matches_both_modalities():
    assert audit_manifests()==dict(train=700,validation=150,test=150)


def test_missing_targets_not_zero_imputed_and_sources_after_cutoff():
    full=pd.read_csv(config.BIOMARKERS_PATH)
    for split in config.SPLITS:
        ds=AuditedTemporalDataset(split)
        assert len(ds)+len(ds.exclusions)==full[full.split.eq(split)].patient_id.nunique()
        for sample in ds:
            assert sample['last_input_day']<=90<sample['target_day']
            assert 108<=sample['target_day']<=132
            source=full[full.patient_id.eq(sample['patient_id'])&full.days_from_baseline.eq(sample['target_day'])].iloc[0]
            assert float(sample['target_ctdna'])==pytest.approx(source.ctDNA_vaf_percent)


def test_future_values_cannot_change_features():
    original=pd.read_csv(config.BIOMARKERS_PATH)
    before=AuditedTemporalDataset('validation')
    modified=original.copy()
    future=modified.days_from_baseline.gt(90)
    cols=config.TEMPORAL_NUMERICAL_FEATURES[:5]
    modified.loc[future,cols]=modified.loc[future,cols]*100+999
    with patch('dl.research_data.pd.read_csv',return_value=modified):
        after=AuditedTemporalDataset('validation')
    assert before.norm_params==after.norm_params
    assert [x['patient_id'] for x in before]==[x['patient_id'] for x in after]
    for x,y in zip(before,after):
        assert torch.equal(x['features'],y['features'])


def test_temporal_raw_days_and_reproducible_challenges():
    for condition in ['missing_visits','short_histories','irregular_intervals','measurement_noise','temporary_spikes','outliers','missing_biomarkers']:
        a=AuditedTemporalDataset('validation',challenge=condition,seed=123)
        b=AuditedTemporalDataset('validation',challenge=condition,seed=123)
        for x,y in zip(a,b):
            assert torch.equal(x['features'],y['features'])
            length=x['length'];days=x['features'][:length,7]
            assert (days<=90).all() and (days>=0).all()
            if length>1:
                assert torch.allclose(x['features'][1:length,6],days.diff(),atol=1e-5)


def test_bag_challenge_nonconstant_and_patient_ownership():
    meta=pd.read_csv(config.IMAGE_METADATA_PATH)
    bags=composition_bags(meta)
    assert set(b['patient_label'] for b in bags)=={0,1,2}
    assert bags==composition_bags(meta)
    for bag in bags:
        assert len(bag['rows'])==12
        assert all(r['patient_id']==bag['patient_id'] for r in bag['rows'])


def test_pretrained_download_failure_is_not_random_fallback():
    with patch.object(pb.models,'resnet50',side_effect=RuntimeError('download unavailable')):
        with pytest.raises(RuntimeError,match='download unavailable'):
            pb.ResNet50Transfer(pretrained=True,freeze_backbone=True)


def test_random_backbone_remains_trainable():
    model=pb.ResNet18Transfer(pretrained=False,freeze_backbone=True)
    assert model.backbone.conv1.weight.requires_grad


def test_undefined_metrics_not_invented():
    m=classification(np.zeros(5),np.arange(5)/5)
    assert np.isnan(m['roc_auc']) and np.isnan(m['pr_auc'])
    with pytest.raises(ValueError):classification([0,1],[np.nan,.8])


def test_multiclass_pr_auc_is_measured():
    m=classification(np.array([0,1,2,0,1,2]),np.eye(3)[[0,1,2,0,1,2]])
    assert m['pr_auc']==pytest.approx(1.)


def test_negative_degradation_and_zero_reference():
    assert degradation(.5,.75)==(-.25,-50.)
    assert np.isnan(degradation(0.,.1)[1])


def test_mc_binary_probabilities_independent_of_batch_and_modes_restored():
    net=nn.Sequential(nn.Linear(3,1),nn.Dropout(.5),nn.Flatten(0))
    net.eval()
    before=[m.training for m in net.modules()]
    result=MCDropoutEvaluator(30).evaluate_classification(net,torch.zeros(4,3))
    assert [m.training for m in net.modules()]==before
    assert result['all_samples'].shape==(30,4)
    assert not np.allclose(result['all_samples'].sum(1),1.)


def test_mil_attention_receives_gradient_and_is_permutation_invariant():
    torch.manual_seed(42)
    model=CachedMIL(2048,'attention');x=torch.randn(2,12,2048)
    logits,_,_=model(x);logits.sum().backward()
    assert model.attention.attention_w.weight.grad.abs().sum()>0
    model.eval()
    with torch.no_grad():
        a,_,w=model(x);b,_,_=model(x[:,torch.randperm(12)])
    assert torch.allclose(a,b,atol=1e-6)
    assert torch.allclose(w.sum(1),torch.ones(2))


def test_transformer_padding_invariance_and_time_sensitivity():
    torch.manual_seed(42)
    model=ContinuousTemporalTransformer().eval()
    x=torch.randn(2,10,13);lengths=torch.tensor([3,4])
    y=x.clone()
    for i,n in enumerate(lengths):y[i,n:]=9999
    with torch.no_grad():
        a=model(x,lengths);b=model(y,lengths)
        z=x.clone();z[:,:,7]+=15;c=model(z,lengths)
    assert torch.allclose(a[1],b[1],atol=1e-6)
    assert not torch.allclose(a[1],c[1])
