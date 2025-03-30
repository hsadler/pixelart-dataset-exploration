from abc import ABC, abstractmethod
import torch


# TODO: maybe use this to abstract the model configurations
class Model(ABC):
    def __init__(self, model_path: str, device: str, image_pixel_size: int):
        self.model_path = model_path
        self.device = device
        self.image_pixel_size = image_pixel_size

    @classmethod
    def new(cls, model_type: str, device: str, image_pixel_size: int) -> "Model":
        model_path = f"models/{model_type}.pth"
        return cls(model_path, device, image_pixel_size)

    @classmethod
    def from_path(cls, model_path: str, device: str, image_pixel_size: int) -> "Model":
        return cls(model_path, device, image_pixel_size)

    @abstractmethod
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        pass
