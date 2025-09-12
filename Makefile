.PHONY: setup test lint clean run

setup:
	pip install -e .

test:
	python -m pytest tests/

lint:
	black .
	isort .
	flake8 .

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +

run:
	python src/main.py
