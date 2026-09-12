"""Small checkpoint helpers for Q7 training and inference."""

from pathlib import Path

import torch


def checkpoint_path(output_directory: str | Path, choice: str) -> Path:
    """Return the path of the selected best/latest checkpoint."""
    output_path = Path(output_directory)

    if choice == "best":
        return output_path / "best_checkpoint.pth"
    if choice == "latest":
        return output_path / "latest_checkpoint.pth"

    raise ValueError("checkpoint choice must be 'best' or 'latest'")


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    train_losses: list[float],
    validation_losses: list[float],
    best_validation_loss: float,
    epochs_without_improvement: int,
) -> None:
    """Save exactly the state needed to continue training correctly."""
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_losses": train_losses,
            "validation_losses": validation_losses,
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
        },
        Path(path),
    )


def load_checkpoint(path, model, device, optimizer=None, scheduler=None):
    """Load model state and, when supplied, the training state as well."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint
