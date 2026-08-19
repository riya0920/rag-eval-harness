.PHONY: test matrix gate baseline demo generation judge
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
generation:
	PYTHONPATH=src python -m ragkit.generation_eval run
judge:
	PYTHONPATH=src python -m ragkit.generation_eval validate-judge
