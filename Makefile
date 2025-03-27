# Constants
MODEL_TYPE_HIGH_CAPACITY = high_capacity
MODEL_TYPE_LOW_CAPACITY = low_capacity
IMAGE_SIZE_SMALL = 32
IMAGE_SIZE_MEDIUM = 64
BATCH_SIZE_OVERFIT = 5
BATCH_SIZE_TRAIN = 50
EPOCHS_SMALL = 50
EPOCHS_MEDIUM = 150
EPOCHS_LARGE = 300
DEVICE = mps

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
		--model-type=$(MODEL_TYPE_HIGH_CAPACITY) \
		--batch-size=$(BATCH_SIZE_TRAIN) \
		--num-epochs=$(EPOCHS_SMALL) \
		--image-pixel-size=$(IMAGE_SIZE_SMALL) \
		--device=$(DEVICE)
