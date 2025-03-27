from enum import Enum
import torch


class ModelType(Enum):
    HIGH_CAPACITY = "high_capacity"
    LOW_CAPACITY = "low_capacity"


def new_model(model_type: ModelType, image_pixel_size: int) -> torch.nn.Sequential:
    high_capacity_model: torch.nn.Sequential = torch.nn.Sequential(
        # Encoder
        torch.nn.Flatten(),
        torch.nn.Linear(3 * image_pixel_size * image_pixel_size, 1024),  # Increased capacity
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),  # Added another layer
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),  # Latent representation
        torch.nn.ReLU(),
        # Decoder (mirror the encoder)
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 3 * image_pixel_size * image_pixel_size),
    )
    low_capacity_model: torch.nn.Sequential = torch.nn.Sequential(
        # Encoder
        torch.nn.Flatten(),
        torch.nn.Linear(3 * image_pixel_size * image_pixel_size, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),  # This creates the latent representation
        torch.nn.ReLU(),
        # Decoder
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 3 * image_pixel_size * image_pixel_size),
    )
    return {
        ModelType.HIGH_CAPACITY: high_capacity_model,
        ModelType.LOW_CAPACITY: low_capacity_model,
    }[model_type]


def predict(
    model_type: ModelType,
    model_path: str,
    input_tensor: torch.Tensor,
    image_pixel_size: int,
    device: str,
) -> torch.Tensor:
    # Load model
    model: torch.nn.Sequential = new_model(model_type=model_type, image_pixel_size=image_pixel_size)
    model = model.to(device)  # Move model to device
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load state dict with correct device mapping
    model.eval()
    # Predict
    input_tensor: torch.Tensor = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs: torch.Tensor = model(input_tensor)
        outputs: torch.Tensor = outputs.view(outputs.size(0), 3, image_pixel_size, image_pixel_size)
        outputs: torch.Tensor = outputs.clamp(0, 1)  # Ensure values are between 0 and 1
        outputs: torch.Tensor = outputs.cpu()
        outputs: torch.Tensor = outputs.squeeze(0)  # Remove batch dimension
    return outputs


def predict_from_dataset_index(
    model_type: ModelType,
    model_path: str,
    ds: torch.utils.data.Dataset,
    ds_index: int,
    image_pixel_size: int,
    device: str,
) -> torch.Tensor:
    input_tensor: torch.Tensor = ds[ds_index]["tensor"]
    output_tensor: torch.Tensor = predict(
        model_type=model_type,
        model_path=model_path,
        input_tensor=input_tensor,
        image_pixel_size=image_pixel_size,
        device=device,
    )
    return output_tensor
