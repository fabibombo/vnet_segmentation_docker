import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def save_model(
    model: torch.nn.Module, target_dir: str, model_name: str, weights: bool = True
):
    """Saves a PyTorch model to a target directory.

    Args:
      model: A target PyTorch model to save.
      target_dir: A directory for saving the model to.
      model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.
      weights: Save the model with (T) or without (F) weights.

    Example usage:
      save_model(model=model_0,
                 target_dir="models",
                 model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    if weights:
        torch.save(obj=model, f=model_save_path)
    else:
        torch.save(obj=model.state_dict(), f=model_save_path)


def set_seeds(seed: int = 42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def plot_all_frames(image, mask, prediction_model=None, threshold=0.5, nll=False):
    num_frames = image.shape[-1]
    num_rows = 2

    if prediction_model is not None:
        num_rows = 3
        predictions = prediction_model(image.unsqueeze(dim=0))

        if nll:
            predictions = F.softmax(predictions, dim=1)
            predictions = predictions[:,1]
        else:
            predictions = predictions[:,1]
            
        predictions = predictions.view(mask.unsqueeze(dim=0).shape)
        predictions = predictions >= threshold
        predictions = predictions.squeeze(dim=0).cpu().detach().numpy()

    fig, axes = plt.subplots(
        num_rows, num_frames, figsize=(3 * num_frames, 3 * num_rows)
    )

    for i in range(num_frames):
        # Plot the image in black and white without axis
        axes[0, i].imshow(
            image[0, :, :, i], cmap="gray"
        )  # Assuming the first channel is the only one
        axes[0, i].set_title(f"Frame {i}")
        axes[0, i].axis("off")

        # Plot the mask in black and white without axis
        axes[1, i].imshow(
            mask[0, :, :, i], cmap="gray"
        )  # Assuming the first channel is the only one
        axes[1, i].set_title(f"Mask {i}")
        axes[1, i].axis("off")

        # Plot predictions if provided
        if prediction_model is not None:
            axes[2, i].imshow(
                predictions[0, :, :, i], cmap="gray"
            )  # Assuming the first channel is the only one
            axes[2, i].set_title(f"Prediction {i}")
            axes[2, i].axis("off")

    plt.tight_layout()
    plt.show()
