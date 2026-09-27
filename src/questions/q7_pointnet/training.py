"""Simple and complete Q7 training loop."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from questions.q7_pointnet import config
from questions.q7_pointnet.checkpoint import (
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from questions.q7_pointnet.model import PointNetDenoiser
from questions.q7_pointnet.monitoring import (
    create_tensorboard_writer,
    log_epoch,
    save_training_curves,
)


def train_one_epoch(
    model: PointNetDenoiser,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for patches, targets in loader:
        patches = patches.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(patches)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(patches)

    return total_loss / len(loader.dataset)


def evaluate(
    model: PointNetDenoiser,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for patches, targets in loader:
            patches = patches.to(device)
            targets = targets.to(device)

            predictions = model(patches)
            loss = loss_function(predictions, targets)
            total_loss += loss.item() * len(patches)

    return total_loss / len(loader.dataset)


def train_model(
    train_dataset,
    validation_dataset,
    output_directory: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    resume_from: str | None = None,
):
    """Train, validate, schedule LR, early-stop, and save best/latest states."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    best_path = output_path / "best_checkpoint.pth"
    latest_path = output_path / "latest_checkpoint.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = PointNetDenoiser().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Our useful runs tend to settle well before 20 epochs. StepLR is simple
    # and predictable: train 10 full epochs at the initial rate, then halve it.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.SCHEDULER_STEP_SIZE,
        gamma=config.SCHEDULER_GAMMA,
    )

    start_epoch = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = np.inf
    epochs_without_improvement = 0

    # Optional resume from a complete best/latest training checkpoint.
    if resume_from is not None:
        path = checkpoint_path(output_directory, resume_from)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = load_checkpoint(
            path,
            model,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = checkpoint["epoch"] + 1
        train_losses = checkpoint["train_losses"]
        validation_losses = checkpoint["validation_losses"]
        best_validation_loss = checkpoint["best_validation_loss"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        print(f"Resuming from {resume_from} checkpoint at epoch {start_epoch}.")

    writer = create_tensorboard_writer(output_directory)

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
        )
        validation_loss = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
        )

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            is_best = True
        else:
            epochs_without_improvement += 1
            is_best = False

        # Learning rate used for the epoch that just finished.
        learning_rate_now = optimizer.param_groups[0]["lr"]

        # Prepare the learning rate for the next epoch. The checkpoint stores
        # this updated scheduler/optimizer state, so resume remains correct.
        scheduler.step()

        # Save latest every epoch so interrupted training can continue.
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            epoch,
            train_losses,
            validation_losses,
            best_validation_loss,
            epochs_without_improvement,
        )

        # Best is the checkpoint used for final inference/evaluation.
        if is_best:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                train_losses,
                validation_losses,
                best_validation_loss,
                epochs_without_improvement,
            )
            print("New best model saved.")

        log_epoch(
            writer,
            epoch,
            train_loss,
            validation_loss,
            learning_rate_now,
        )
        save_training_curves(
            train_losses,
            validation_losses,
            output_path / "training_curves.png",
        )

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"train={train_loss:.6f} | "
            f"val={validation_loss:.6f} | "
            f"best={best_validation_loss:.6f} | "
            f"lr={learning_rate_now:.2e}"
        )

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print(
                "Early stopping: validation loss did not improve for "
                f"{config.EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    writer.close()
    save_training_curves(
        train_losses,
        validation_losses,
        output_path / "training_curves.png",
    )

    return model, train_losses, validation_losses
