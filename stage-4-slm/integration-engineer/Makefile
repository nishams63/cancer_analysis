# Integration Engineer Automation Makefile

.PHONY: install merge quantize validate benchmark test serve offline-check

install:
	pip install -r requirements.txt

merge:
	python scripts/merge_lora.py

quantize:
	python scripts/convert_and_quantize.py

validate:
	python scripts/validate_quantization.py

benchmark:
	python scripts/run_performance_benchmark.py

test:
	pytest tests/ -v

offline-check:
	python scripts/offline_validation.py

serve:
	uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
