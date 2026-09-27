"""TensorBoard logging and the final train/validation loss plot."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def create_tensorboard_writer(output_directory: str | Path) -> SummaryWriter:
    log_directory = Path(output_directory) / "tensorboard"
    return SummaryWriter(log_dir=str(log_directory))


def log_epoch(
    writer: SummaryWriter,
    epoch: int,
    train_loss: float,
    validation_loss: float,
    learning_rate: float,
) -> None:
    """Log only the quantities we actually use to understand training."""
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Validation", validation_loss, epoch)
    writer.add_scalar("Training/Learning_Rate", learning_rate, epoch)


def save_training_curves(
    train_losses: list[float],
    validation_losses: list[float],
    output_path: str | Path,
) -> None:
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Training loss")
    plt.plot(epochs, validation_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("PointNet training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
