from fire import Fire
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2, ToPILImage
from PIL import Image
from tqdm import tqdm


# HELPER FUNCTIONS


def _load_and_preprocess_dataset(
    path:str,
    split:str,
    image_pixel_size:int,
) -> DataLoader:
    ds = load_dataset(path, split=split)
    transform = v2.Compose([
        v2.Lambda(lambda x: x.convert("RGB")),
        v2.Lambda(lambda x: _scale_image_by_pixel_size(x, image_pixel_size)),
        v2.CenterCrop((image_pixel_size, image_pixel_size)),
        v2.ToTensor(),
    ])
    def preprocess(examples):
        tensors = [transform(image) for image in examples["image"]]
        return {"tensor": torch.stack(tensors)}
    ds = ds.map(preprocess, batched=True)
    ds.set_format(type="torch", columns=["tensor"])
    return ds


def _scale_image_by_pixel_size(image: Image.Image, pixel_size: int) -> Image.Image | None:
    original_width, original_height = image.size
    if original_width <= original_height:
        # scale by width
        scale_factor = original_width / pixel_size
        new_width = pixel_size
        new_height = int(original_height / scale_factor)
    else:
        # scale by height
        scale_factor = original_height / pixel_size
        new_height = pixel_size
        new_width = int(original_width / scale_factor)
    scaled_image = image.resize((new_width, new_height), resample=Image.Resampling.NEAREST)
    return scaled_image


def _instantiate_model(image_pixel_size: int = 64) -> torch.nn.Sequential:
    model: torch.nn.Sequential = torch.nn.Sequential(
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
    return model


# CLI COMMANDS


def create_test_image(image_pixel_size: int = 64):
    ds = _load_and_preprocess_dataset(
        path="tkarr/sprite_caption_dataset",
        split="train",
        image_pixel_size=image_pixel_size,
    )
    # get first image and save it
    t: torch.Tensor = ds[0]["tensor"]
    pil_image: Image.Image = ToPILImage()(t)
    pil_image.save("test_image.png")


def train(
    batch_size: int = 16,
    image_pixel_size: int = 64,
    num_epochs: int = 10,
    overfit: bool = False,
):
    config_dict = {k: v for k, v in locals().items()}
    print("Config:")
    for k, v in config_dict.items():
        print(f"  {k}: {v}")
    
    # Load and preprocess the datasets
    print("Loading and preprocessing datasets...")
    train_ds: torch.utils.data.Dataset = _load_and_preprocess_dataset(
        path="tkarr/sprite_caption_dataset",
        split="train",
        image_pixel_size=image_pixel_size,
    )
    val_ds: torch.utils.data.Dataset = _load_and_preprocess_dataset(
        path="tkarr/sprite_caption_dataset",
        split="valid",
        image_pixel_size=image_pixel_size,
    )
    if overfit:
        train_ds = train_ds.select(range(10))
        val_ds = train_ds
    
    print("Creating data loaders...")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Train a simple autoencoder
    print("Instantiating model...")
    model = _instantiate_model(image_pixel_size=image_pixel_size)
    
    print("Defining loss function and optimizer...")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training the model...")
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):
            optimizer.zero_grad()
            inputs = batch["tensor"]
            inputs_flat = inputs.view(inputs.size(0), -1)  # Flatten to (batch_size, channels*height*width)
            outputs = model(inputs_flat)
            loss = criterion(outputs, inputs_flat)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["tensor"]
                inputs_flat = inputs.view(inputs.size(0), -1)  # Flatten to (batch_size, channels*height*width)
                outputs = model(inputs_flat)
                loss = criterion(outputs, inputs_flat)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")


def predict(
    model_path: str,
    input_tensor: torch.Tensor,
    image_pixel_size: int,
) -> torch.Tensor:
    # Load model
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: torch.nn.Sequential = _instantiate_model(image_pixel_size=image_pixel_size)
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


def predict_from_image(
    model_path: str = "best_model.pth",
    image_path: str = "test_image.png",
    output_path: str = "output_image.png",
    image_pixel_size: int = 64,
) -> None:
    image: Image.Image = Image.open(image_path)
    image: Image.Image = _scale_image_by_pixel_size(image, image_pixel_size)
    input_tensor: torch.Tensor = v2.ToTensor()(image)
    output_tensor: torch.Tensor = predict(model_path, input_tensor, image_pixel_size)
    output_image: Image.Image = ToPILImage()(output_tensor)
    output_image.save(output_path)


def predict_from_dataset_index(
    model_path: str = "best_model.pth",
    ds: torch.utils.data.Dataset = None,
    ds_index: int = 0,
    image_pixel_size: int = 64,
) -> torch.Tensor:
    input_tensor: torch.Tensor = ds[ds_index]["tensor"]
    output_tensor: torch.Tensor = predict(model_path, input_tensor, image_pixel_size)
    return output_tensor


if __name__ == "__main__":
    Fire({
        "create_test_image": create_test_image,
        "train": train,
        "predict_from_image": predict_from_image,
    })
