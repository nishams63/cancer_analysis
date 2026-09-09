import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from protocol import span_metrics, load_development, digest

def test_locked_test_denied():
    with pytest.raises(ValueError, match='forbidden'): load_development('LOCKED_TEST')

def test_duplicate_prediction_counts_as_false_positive():
    g={'start':0,'end':3,'label':'DRUG_NAME'}
    m=span_metrics([[g]],[[g,g]])['micro']
    assert m['tp']==1 and m['precision']==.5 and m['recall']==1

def test_relaxed_matching_not_greedy():
    g=[{'start':0,'end':2,'label':'DOSAGE'},{'start':2,'end':4,'label':'DOSAGE'}]
    p=[{'start':0,'end':4,'label':'DOSAGE'},{'start':0,'end':2,'label':'DOSAGE'}]
    assert span_metrics([g],[p],True)['micro']['tp']==2
    assert span_metrics([g],[p],False)['micro']['tp']==1

def test_no_identifier_in_hash():
    assert len(digest('internal-patient-id')) == 64
