
create-test-image:
	poetry run python -m cli create_test_image

train:
	poetry run python -m cli train

train-overfit:
	poetry run python -m cli train --overfit=True --batch-size=5 --num-epochs=100

predict:
	poetry run python -m cli predict
