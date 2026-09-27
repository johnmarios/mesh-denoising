"""Final Q7 comparison against the analytical denoising methods."""

import csv
from pathlib import Path

import numpy as np
import torch

from core.mesh import copy_mesh, create_mesh, prepare_mesh, vertex_normals
from questions import q1_noise, q2_metrics, q4_smoothing, q5_feature_denoising
from questions.q7_pointnet import config
from questions.q7_pointnet.inference import denoise_mesh_with_model, load_trained_model


METHODS = ("Noisy", "Laplacian", "Taubin", "FeatureAware", "PointNet")


def _mesh_with_vertices(reference_mesh, vertices):
    """Copy connectivity from reference_mesh and replace only vertex positions."""
    result = copy_mesh(reference_mesh)
    result.vertices = np.asarray(vertices, dtype=float)
    result.vertex_normals = vertex_normals(result.vertices, result.triangles)
    return result


def evaluate_pointnet(output_directory: str | Path = config.OUTPUT_DIR) -> Path:
    """Evaluate Q7 on held-out meshes and save one easy-to-read CSV file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(
        str(output_directory),
        config.INFERENCE_CHECKPOINT,
        device,
    )

    rows = []
    case_index = 0

    for model_name in config.TEST_MODELS:
        clean_mesh = prepare_mesh(create_mesh(model_name))
        clean_vertices = np.asarray(clean_mesh.vertices, dtype=float)
        triangles = np.asarray(clean_mesh.triangles, dtype=np.int64)
        normals = np.asarray(clean_mesh.vertex_normals, dtype=float)

        for noise_type in config.NOISE_TYPES:
            for noise_level in config.NOISE_LEVELS:
                noisy_vertices = q1_noise.add_noise(
                    clean_vertices,
                    triangles,
                    normals,
                    noise_type,
                    noise_level,
                    seed=config.SEED + case_index,
                )
                case_index += 1
                noisy_mesh = _mesh_with_vertices(clean_mesh, noisy_vertices)

                laplacian_vertices = q4_smoothing.laplacian_smoothing(
                    noisy_mesh,
                    strength=config.EVALUATION_STRENGTH,
                    iterations=config.EVALUATION_ITERATIONS,
                )
                taubin_vertices = q4_smoothing.taubin_smoothing(
                    noisy_mesh,
                    strength=config.EVALUATION_STRENGTH,
                    iterations=config.EVALUATION_ITERATIONS,
                )
                feature_vertices, _ = q5_feature_denoising.feature_aware_denoising(
                    noisy_vertices,
                    triangles,
                    strength=config.EVALUATION_STRENGTH,
                    iterations=config.EVALUATION_ITERATIONS,
                )
                pointnet_vertices = denoise_mesh_with_model(
                    noisy_mesh,
                    model,
                    device,
                    k=config.PATCH_SIZE,
                )

                results = {
                    "Noisy": noisy_vertices,
                    "Laplacian": laplacian_vertices,
                    "Taubin": taubin_vertices,
                    "FeatureAware": feature_vertices,
                    "PointNet": pointnet_vertices,
                }

                print(
                    f"\n{model_name} | {noise_type} | level={noise_level:.2f}"
                )
                print("Method          RMSE         MSAE(deg)    HF ratio")
                print("------------------------------------------------------")

                for method in METHODS:
                    current_mesh = _mesh_with_vertices(clean_mesh, results[method])
                    metrics = q2_metrics.compute_metrics(clean_mesh, current_mesh)

                    row = {
                        "model": model_name,
                        "noise_type": noise_type,
                        "noise_level": noise_level,
                        "method": method,
                    }
                    row.update(metrics)
                    rows.append(row)

                    print(
                        f"{method:14s} "
                        f"{metrics['RMSE']:<12.6g} "
                        f"{metrics['MSAE_deg']:<12.6g} "
                        f"{metrics['HF_ratio']:<12.6g}"
                    )

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "q7_results.csv"

    fieldnames = [
        "model",
        "noise_type",
        "noise_level",
        "method",
        "MSAE_deg",
        "L2_surface",
        "Hausdorff",
        "RMSE",
        "HF_ratio",
        "Radius_ratio",
        "AABB_ratio",
        "Area_ratio",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nQ7 evaluation saved to: {csv_path}")
    return csv_path
