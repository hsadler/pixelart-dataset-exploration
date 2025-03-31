from fire import Fire
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from PIL import Image
from tqdm import tqdm
import wandb
from wandb.sdk.wandb_run import Run

from vae import ModelType, new_model, predict, train_predict
from utils import pil_image_concat, tensor_to_pil_image


# HELPER FUNCTIONS


def _load_and_preprocess_dataset(
    path: str,
    split: str,
    image_pixel_size: int,
) -> DataLoader:
    ds = load_dataset(path, split=split)
    transform = v2.Compose(
        [
            v2.Lambda(lambda x: x.convert("RGB")),
            # TODO maybe convert to HSL?
            # v2.Lambda(lambda x: x.convert("HSL")),
            v2.Lambda(lambda x: _scale_image_by_pixel_size(x, image_pixel_size)),
            v2.CenterCrop((image_pixel_size, image_pixel_size)),
            v2.ToTensor(),
        ]
    )

    def preprocess(examples):
        tensors = [transform(image) for image in examples["image"]]
        return {"tensor": torch.stack(tensors)}

    ds = ds.map(preprocess, batched=True)
    ds.set_format(type="torch", columns=["tensor"])
    return ds


def _scale_image_by_pixel_size(image: Image.Image, pixel_size: int) -> Image.Image:
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


# CLI COMMANDS


def create_test_image(image_pixel_size: int):
    ds = _load_and_preprocess_dataset(
        path="tkarr/sprite_caption_dataset",
        split="train",
        image_pixel_size=image_pixel_size,
    )
    # get first image and save it
    t: torch.Tensor = ds[0]["tensor"]
    pil_image: Image.Image = tensor_to_pil_image(t)
    pil_image.save("test_image.png")


def train(
    model_type: str,
    image_pixel_size: int,
    batch_size: int,
    num_epochs: int,
    device: str,
    overfit: bool = False,
    wandb_project: str = "pixelart-autoencoder",
):
    config_dict = {k: v for k, v in locals().items()}

    # Initialize wandb
    print("Initializing wandb...")
    run: Run = wandb.init(
        project=wandb_project,
        config=config_dict,
    )

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
        train_ds = train_ds.select(range(100))
        val_ds = train_ds

    print("Creating data loaders...")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Select 10 fixed images for visualization
    viz_batch = next(iter(val_loader))
    viz_inputs: torch.Tensor = viz_batch["tensor"][:10].to(device)

    print("Instantiating model...")
    model = new_model(model_type=ModelType(model_type), image_pixel_size=image_pixel_size)
    model = model.to(device)

    print("Defining loss function and optimizer...")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training the model...")
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):
            optimizer.zero_grad()
            inputs = batch["tensor"].to(device)
            outputs: torch.Tensor = train_predict(model, inputs, no_grad=False)
            loss: torch.Tensor = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["tensor"].to(device)
                outputs: torch.Tensor = train_predict(model, inputs, no_grad=True)
                loss: torch.Tensor = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader)

        # Log sample input and reconstruction images side by side in a single grid
        imgs: list[Image.Image] = []
        if epoch % 10 == 0 or True:
            viz_outputs: torch.Tensor = train_predict(model, viz_inputs, no_grad=True)
            # Create pairs of original and reconstructed images
            for idx in range(5):
                # Convert tensors to PIL images and append
                imgs.append(tensor_to_pil_image(viz_inputs[idx]))
                imgs.append(tensor_to_pil_image(viz_outputs[idx]))
            run.log(
                {
                    "comparisons": wandb.Image(
                        pil_image_concat(imgs),
                        mode="RGB",
                        caption="Left: Original, Right: Reconstruction",
                    )
                }
            )

        # Log metrics and images to wandb
        run.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            # Log best model to wandb
            print(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")

    # Log details
    print("Training complete!")
    print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print("Config:")
    for k, v in config_dict.items():
        print(f"  {k}: {v}")

    # Close wandb run
    run.finish()


def predict_from_image(
    model_type: str,
    model_path: str = "best_model.pth",
    image_path: str = "test_image.png",
    output_path: str = "output_image.png",
    image_pixel_size: int = 64,
    device: str = "cpu",
) -> None:
    image: Image.Image = Image.open(image_path)
    image: Image.Image = _scale_image_by_pixel_size(image, image_pixel_size)
    input_tensor: torch.Tensor = v2.ToTensor()(image)
    output_tensor: torch.Tensor = predict(
        model_type=ModelType(model_type),
        model_path=model_path,
        input_tensor=input_tensor,
        image_pixel_size=image_pixel_size,
        device=device,
    )
    output_image: Image.Image = tensor_to_pil_image(output_tensor)
    output_image.save(output_path)


if __name__ == "__main__":
    Fire(
        {
            "train": train,
            "create_test_image": create_test_image,
            "predict_from_image": predict_from_image,
        }
    )
