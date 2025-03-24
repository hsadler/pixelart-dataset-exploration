
create-test-image:
	poetry run python -m cli create-test-image

train:
	poetry run python -m cli train

train-sample-size-100:
	poetry run python -m cli train --sample_size=100

predict:
	poetry run python -m cli predict
