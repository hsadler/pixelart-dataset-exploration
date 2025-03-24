# CLI for train:
# load the datasets and preps for training
# instantiate the model
# train the model
# save the model

from fire import Fire
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from PIL import Image
import torch
from tqdm import tqdm


def _load_dataset_to_dataloader(
    path:str,
    split:str,
    batch_size:int,
    image_pixel_size:int,
    sample_size:int = None,
) -> DataLoader:
    print("Loading dataset...")
    if sample_size:
        ds = load_dataset(path, split=split).select(range(sample_size))
    else:
        ds = load_dataset(path, split=split)
    
    print("Defining transform...")
    transform = v2.Compose([
        v2.Lambda(lambda x: x.convert("RGB")),
        v2.Lambda(lambda x: _scale_image_by_pixel_size(x, image_pixel_size)),
        v2.CenterCrop((image_pixel_size, image_pixel_size)),
        v2.ToTensor(),
    ])

    def preprocess(examples):
        tensors = [transform(image) for image in examples["image"]]
        return {"tensor": torch.stack(tensors)}

    print("Transforming images...")
    ds = ds.map(preprocess, batched=True)
    ds.set_format(type="torch", columns=["tensor"])

    print("Creating data loader...")
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


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


def train(batch_size: int = 16, image_pixel_size: int = 64, sample_size: int = None):
    config_dict = {k: v for k, v in locals().items()}
    print("Config:")
    for k, v in config_dict.items():
        print(f"  {k}: {v}")
    
    train_loader = _load_dataset_to_dataloader(
        path="tkarr/sprite_caption_dataset",
        split="train",
        batch_size=batch_size,
        image_pixel_size=image_pixel_size,
        sample_size=sample_size,
    )
    val_loader = _load_dataset_to_dataloader(
        path="tkarr/sprite_caption_dataset",
        split="valid",
        batch_size=batch_size,
        image_pixel_size=image_pixel_size,
        sample_size=sample_size,
    )

    # Train a simple autoencoder
    print("Instantiating model...")
    model = torch.nn.Sequential(
        # Encoder
        torch.nn.Flatten(),
        torch.nn.Linear(3 * image_pixel_size * image_pixel_size, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        # Decoder
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 3 * image_pixel_size * image_pixel_size),
    )
    
    print("Defining loss function and optimizer...")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training the model...")
    num_epochs = 10
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


if __name__ == "__main__":
    Fire({
        "train": train,
    })
