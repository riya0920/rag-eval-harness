.PHONY: test matrix gate baseline demo generation judge heldout
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
heldout:
	PYTHONPATH=src python -m ragkit.experiment heldout
judge:
	PYTHONPATH=src python -m ragkit.generation_eval validate-judge
clarify:
	PYTHONPATH=src python -m ragkit.clarify sweep
entity-ambiguity:
	PYTHONPATH=src python -m ragkit.entity_ambiguity
