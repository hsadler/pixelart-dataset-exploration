
train:
	poetry run python train.py

train-sample-size-100:
	poetry run python -m train train --sample_size=100
