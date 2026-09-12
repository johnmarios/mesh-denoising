# Q7 / Q8 clean update

## Q7

The existing dataset/checkpoint/TensorBoard organization is kept.
Only surgical training/model changes were made:

- Batch Normalization was added after hidden Conv/Linear layers, following PointNet.
- The quaternion T-Net starts exactly at the identity rotation.
- The displacement output keeps normal random weights; only its bias is zero.
- `ReduceLROnPlateau` was replaced by a simple `StepLR`.
- `StepLR(step_size=10, gamma=0.5)` is used because previous useful runs were already close to convergence around epoch 16; a 20-epoch step would usually happen too late.
- Early stopping, TensorBoard, best/latest checkpoints and resume are kept.
- Impulse centers are balanced in validation as well as training so early stopping is not dominated by zero targets. Final Q7 full-mesh evaluation remains unbiased.

Because BatchNorm changes the model state, train Q7 from a fresh checkpoint (`RESUME_FROM = None`). Old checkpoints are not compatible with the new architecture.

## Q8

Run a corresponding OBJ sequence with:

```bash
python project.py --dynamic-folder resources/dynamic/<sequence>
```

The dynamic implementation is separated into small files:

- `sequence.py`: loading, correspondence checks, common normalization, noise.
- `spectral.py`: Algorithm 1 from Arvanitis et al. step by step.
- `evaluation.py`: sequence metrics and angular normal-error heatmap.
- `viewer.py`: frame navigation and Q8 controls.
- `config.py`: the few RPCA parameters not numerically specified by the paper.

The implementation performs the paper's full Laplacian eigen-decomposition, so use a reasonably low-resolution sequence for the coursework implementation. Very dense sequences can make the dense eigen-decomposition expensive.
