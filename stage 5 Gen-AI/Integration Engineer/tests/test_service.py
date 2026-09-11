from src.service import Stage5IntegrationService


def test_service_health():
    svc = Stage5IntegrationService()
    h = svc.get_health()
    assert h["status"] == "HEALTHY"
    assert h["version"] == "1.0.0"
