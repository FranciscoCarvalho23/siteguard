.PHONY: install data train evaluate export api test lint docker

install:
	pip install -r requirements.txt
data:
	python scripts/prepare_data.py
train:
	python scripts/train.py
evaluate:
	python scripts/evaluate.py --model models/ppe_yolo11n_640.pt
export:
	python scripts/export_onnx.py --model models/ppe_yolo11n_640.pt
api:
	uvicorn api.main:app --reload --port 8000
test:
	pytest
lint:
	ruff check .
docker:
	docker build -t siteguard . && docker run --rm -p 8000:8000 siteguard
