# Constants
MODEL_TYPE_SIMPLE = simple
MODEL_TYPE_LOW_CAPACITY = low_capacity
MODEL_TYPE_HIGH_CAPACITY = high_capacity
MODEL_TYPE_VERY_HIGH_CAPACITY = very_high_capacity
MODEL_TYPE_BATCH_NORM = batch_norm
MODEL_TYPE_LEAKY_RELU = leaky_relu
MODEL_TYPE_BATCH_NORM_LEAKY = batch_norm_leaky
MODEL_TYPE_CONV = conv

IMAGE_SIZE_SMALL = 32
IMAGE_SIZE_MEDIUM = 64

BATCH_SIZE_OVERFIT = 5
BATCH_SIZE_TRAIN = 50

EPOCHS_SMALL = 50
EPOCHS_MEDIUM = 150
EPOCHS_LARGE = 300

DEVICE = mps

# Test

train-test-simple:
	poetry run python -m cli train \
		--model-type=$(MODEL_TYPE_SIMPLE) \
		--overfit=True \
		--batch-size=$(BATCH_SIZE_OVERFIT) \
		--num-epochs=$(EPOCHS_SMALL) \
		--image-pixel-size=$(IMAGE_SIZE_MEDIUM) \
		--device=$(DEVICE)

# Medium image overfit test

train-medium-image-overfit:
	poetry run python -m cli train \
		--model-type=$(MODEL_TYPE_HIGH_CAPACITY) \
		--overfit=True \
		--batch-size=$(BATCH_SIZE_OVERFIT) \
		--num-epochs=$(EPOCHS_SMALL) \
		--image-pixel-size=$(IMAGE_SIZE_MEDIUM) \
		--device=$(DEVICE)

# Small image overfit test

train-small-image-overfit:
	poetry run python -m cli train \
		--model-type=$(MODEL_TYPE_HIGH_CAPACITY) \
		--overfit=True \
		--batch-size=$(BATCH_SIZE_OVERFIT) \
		--num-epochs=$(EPOCHS_SMALL) \
		--image-pixel-size=$(IMAGE_SIZE_SMALL) \
		--device=$(DEVICE)

# Real training

train:
	poetry run python -m cli train \
		--model-type=$(MODEL_TYPE_CONV) \
		--batch-size=$(BATCH_SIZE_TRAIN) \
		--num-epochs=$(EPOCHS_SMALL) \
		--image-pixel-size=$(IMAGE_SIZE_MEDIUM) \
		--device=$(DEVICE)

# Python

format:
	poetry run black . --line-length=100
