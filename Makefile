# Constants
MODEL_TYPE_SIMPLE = simple
MODEL_TYPE_LOW_CAPACITY = low_capacity
MODEL_TYPE_HIGH_CAPACITY = high_capacity
MODEL_TYPE_VERY_HIGH_CAPACITY = very_high_capacity
MODEL_TYPE_BATCH_NORM = batch_norm
MODEL_TYPE_LEAKY_RELU = leaky_relu
MODEL_TYPE_BATCH_NORM_LEAKY = batch_norm_leaky
MODEL_TYPE_CONV = conv

LOSS_TYPE_MSE = mse
LOSS_TYPE_L1 = l1
LOSS_TYPE_HYBRID = hybrid
LOSS_TYPE_SSIM = ssim

OPTIMIZER_TYPE_ADAM = adam
OPTIMIZER_TYPE_SGD = sgd
OPTIMIZER_TYPE_RMSPROP = rmsprop
OPTIMIZER_TYPE_ADAMW = adamw

LEARNING_RATE_HIGH = 0.001
LEARNING_RATE_MEDIUM = 0.0001
LEARNING_RATE_LOW = 0.00001

IMAGE_SIZE_32 = 32
IMAGE_SIZE_64 = 64
IMAGE_SIZE_128 = 128

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
		--image-pixel-size=$(IMAGE_SIZE_64) \
		--device=$(DEVICE)

# Medium image overfit test

train-medium-image-overfit:
	poetry run python -m cli train \
		--model-type=$(MODEL_TYPE_HIGH_CAPACITY) \
		--overfit=True \
		--batch-size=$(BATCH_SIZE_OVERFIT) \
		--num-epochs=$(EPOCHS_SMALL) \
		--image-pixel-size=$(IMAGE_SIZE_64) \
		--device=$(DEVICE)

# Small image overfit test

train-small-image-overfit:
	poetry run python -m cli train \
		--model-type=$(MODEL_TYPE_HIGH_CAPACITY) \
		--overfit=True \
		--batch-size=$(BATCH_SIZE_OVERFIT) \
		--num-epochs=$(EPOCHS_SMALL) \
		--image-pixel-size=$(IMAGE_SIZE_32) \
		--device=$(DEVICE)

# Real training

train:
	poetry run python -m cli train \
		--model-type=$(MODEL_TYPE_CONV) \
		--loss-type=$(LOSS_TYPE_HYBRID) \
		--optimizer-type=$(OPTIMIZER_TYPE_ADAMW) \
		--learning-rate=$(LEARNING_RATE_HIGH) \
		--image-pixel-size=$(IMAGE_SIZE_64) \
		--batch-size=$(BATCH_SIZE_TRAIN) \
		--num-epochs=$(EPOCHS_MEDIUM) \
		--device=$(DEVICE)

# Python

format:
	poetry run black . --line-length=100
