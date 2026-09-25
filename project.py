"""Main entry point for the mesh-denoising project."""

import argparse

import config
from core.mesh import MODEL_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mesh denoising")
    valid_models = [name.lower() for name in MODEL_NAMES]

    parser.add_argument(
        "--model",
        choices=valid_models,
        default=config.DEFAULT_MODEL.lower(),
        help="select model for the interactive application",
    )
    parser.add_argument(
        "--dynamic-folder",
        type=str,
        default=None,
        help="folder containing corresponding OBJ frames for Question 8",
    )
    parser.add_argument(
        "--train-pointnet",
        action="store_true",
        help="train the Q7 PointNet",
    )
    parser.add_argument(
        "--evaluate-pointnet",
        action="store_true",
        help="compare Q7 PointNet with the analytical denoisers on test meshes",
    )
    return parser


def build_pointnet_dataset(
    model_names: tuple[str, ...],
    samples_per_variant: int,
    training: bool,
    augment_probability: float,
):
    """Combine datasets from several meshes without mixing the mesh splits."""
    from torch.utils.data import ConcatDataset

    from core.mesh import create_mesh, prepare_mesh
    from questions.q7_pointnet import config as pointnet_config
    from questions.q7_pointnet.data import PatchDenoisingDataset

    datasets = []

    for model_index, model_name in enumerate(model_names):
        print(f"Preparing PointNet dataset for {model_name}...")
        clean_mesh = prepare_mesh(create_mesh(model_name))

        dataset = PatchDenoisingDataset(
            clean_mesh=clean_mesh,
            noise_types=pointnet_config.NOISE_TYPES,
            noise_levels=pointnet_config.NOISE_LEVELS,
            samples_per_variant=samples_per_variant,
            k=pointnet_config.PATCH_SIZE,
            seed=pointnet_config.SEED + 10_000 * model_index,
            augment=training,
            augment_probability=augment_probability,
            # Balance impulse centers in train AND validation so the validation
            # loss used for model selection does not consist mostly of zeros.
            # Final full-mesh evaluation remains unbiased.
            balance_impulse=True,
        )
        datasets.append(dataset)
        print(f"  {len(dataset):,} samples")

    return ConcatDataset(datasets)


def run_pointnet_training() -> None:
    import numpy as np
    import torch

    from questions.q7_pointnet import config as pointnet_config
    from questions.q7_pointnet.training import train_model

    np.random.seed(pointnet_config.SEED)
    torch.manual_seed(pointnet_config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(pointnet_config.SEED)

    print("Building PointNet training dataset...")
    train_dataset = build_pointnet_dataset(
        pointnet_config.TRAIN_MODELS,
        pointnet_config.TRAIN_SAMPLES_PER_VARIANT,
        training=True,
        augment_probability=pointnet_config.ROTATION_AUGMENTATION_PROBABILITY,
    )

    print("Building PointNet validation dataset...")
    validation_dataset = build_pointnet_dataset(
        pointnet_config.VALIDATION_MODELS,
        pointnet_config.VALIDATION_SAMPLES_PER_VARIANT,
        training=False,
        augment_probability=pointnet_config.ROTATION_AUGMENTATION_PROBABILITY,
    )

    print(
        f"Starting training with {len(train_dataset):,} train and "
        f"{len(validation_dataset):,} validation samples."
    )

    train_model(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        output_directory=str(pointnet_config.OUTPUT_DIR),
        epochs=pointnet_config.EPOCHS,
        batch_size=pointnet_config.BATCH_SIZE,
        learning_rate=pointnet_config.LEARNING_RATE,
        resume_from=pointnet_config.RESUME_FROM,
    )


def run_pointnet_evaluation() -> None:
    from questions.q7_pointnet.evaluation import evaluate_pointnet

    evaluate_pointnet()


def main() -> None:
    args = build_parser().parse_args()

    if args.dynamic_folder is not None:
        from questions.q8_dynamic.viewer import DynamicMeshApp

        app = DynamicMeshApp(args.dynamic_folder)
        app.mainLoop()
        return

    if args.train_pointnet:
        run_pointnet_training()
        return

    if args.evaluate_pointnet:
        run_pointnet_evaluation()
        return

    from app import ProjectApp

    app = ProjectApp(args.model.upper())
    app.mainLoop()


if __name__ == "__main__":
    main()
