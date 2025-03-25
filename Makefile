
# Overfit tests

create-test-image-overfit:
	poetry run python -m cli create_test_image --image-pixel-size=64

train-overfit:
	poetry run python -m cli train --overfit=True --batch-size=5 --num-epochs=150 --image-pixel-size=64

predict-overfit:
	poetry run python -m cli predict_from_image --image-pixel-size=64

run-overfit: create-test-image-overfit train-overfit predict-overfit

# Small image overfit tests

create-test-image-small-overfit:
	poetry run python -m cli create_test_image --image-pixel-size=32

train-small-image-overfit:
	poetry run python -m cli train --overfit=True --batch-size=5 --num-epochs=50 --image-pixel-size=32

predict-small-image-overfit:
	poetry run python -m cli predict_from_image --image-pixel-size=32

run-small-image-overfit: create-test-image-small-overfit train-small-image-overfit predict-small-image-overfit


# Large image overfit tests

train-large-image-overfit:
	poetry run python -m cli train --overfit=True --batch-size=5 --num-epochs=50 --image-pixel-size=128

# Real training

create-test-image:
	poetry run python -m cli create_test_image --image-pixel-size=32

train:
	poetry run python -m cli train --batch-size=50 --num-epochs=10 --image-pixel-size=32

predict:
	poetry run python -m cli predict_from_image --image-pixel-size=32

run-real-training: create-test-image train predict
