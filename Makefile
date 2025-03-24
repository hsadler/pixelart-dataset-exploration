
create-test-image:
	poetry run python -m cli create-test-image

train:
	poetry run python -m cli train

train-overfit:
	poetry run python -m cli train --batch-size=5 --subset=10 --num-epochs=300

predict:
	poetry run python -m cli predict
