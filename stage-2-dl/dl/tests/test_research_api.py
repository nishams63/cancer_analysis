from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.runtime'))
sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from integration.src import api
from dl.research_inference import ResearchPredictor,modality_status


def test_all_modality_states():
    assert modality_status(True,True)=='FULL_MULTIMODAL'
    assert modality_status(True,False)=='PATHOLOGY_ONLY'
    assert modality_status(False,True)=='TEMPORAL_ONLY'
    assert modality_status(False,False)=='INSUFFICIENT_DATA'


def test_validated_request_does_not_invent_untrained_predictions():
    with patch.object(ResearchPredictor,'ready',return_value=False):
        response=TestClient(api.app).post('/predict/patient',json={'patient_id':'PAT-test','model_version':'validated'})
    assert response.status_code==503


def test_future_api_visit_is_rejected():
    response=TestClient(api.app).post('/predict/patient',json={'patient_id':'PAT-test','temporal_observations':[{'days_from_baseline':91}]})
    assert response.status_code==422


def test_batch_keeps_per_patient_errors():
    with patch.object(ResearchPredictor,'ready',return_value=False):
        response=TestClient(api.app).post('/predict/batch',json={'patients':[{'patient_id':'a','model_version':'validated'},{'patient_id':'b','model_version':'validated'}]})
    assert response.status_code==200
    assert response.json()['count']==2
    assert all(item['status_code']==503 for item in response.json()['results'])


def test_no_input_abstains_without_loading_models():
    predictor=ResearchPredictor.__new__(ResearchPredictor)
    result=predictor.predict('PAT-empty')
    assert result['modality_status']=='INSUFFICIENT_DATA'
    assert result['progression_probability'] is None
    assert result['ctdna_forecast_30d'] is None
    assert result['uncertainty'] is None


def test_patient_and_health_use_ready_artifacts():
    class Predictor:
        selected='test_validated_configuration'
        def predict(self,*args):return {'patient_id':args[0],'modality_status':'INSUFFICIENT_DATA','progression_probability':None}
    with patch.object(ResearchPredictor,'ready',return_value=True),patch.object(ResearchPredictor,'shared',return_value=Predictor()):
        client=TestClient(api.app)
        assert client.get('/health').json()['selected_configuration']=='test_validated_configuration'
        assert client.post('/predict/patient',json={'patient_id':'PAT-test'}).json()['modality_status']=='INSUFFICIENT_DATA'
