.PHONY: test matrix gate baseline demo
test:
	pytest
matrix:
	PYTHONPATH=src python -m ragkit.experiment matrix
gate:
	PYTHONPATH=src python -m ragkit.experiment gate
baseline:
	PYTHONPATH=src python -m ragkit.experiment update-baseline
demo:
	-PYTHONPATH=src python -m ragkit.experiment gate --plant-regression
