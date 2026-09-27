import torch
from torch import nn
import torch.nn.functional as F


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions (B,4) to rotation matrices (B,3,3)."""

    quaternion = F.normalize(quaternion, p=2, dim=1)

    w = quaternion[:, 0]
    x = quaternion[:, 1]
    y = quaternion[:, 2]
    z = quaternion[:, 3]

    rotation = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),

            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),

            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=1,
    )

    return rotation.reshape(-1, 3, 3)


class QuaternionInputTNet(nn.Module):
    """PointNet-style input alignment, parameterized by a unit quaternion."""

    def __init__(self):
        super().__init__()

        # PointNet uses BatchNorm after the hidden layers. It stabilizes the
        # feature scales seen by the following layers during training.
        self.point_mlp = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.quaternion_output = nn.Linear(64, 4)

        # The PointNet transformation starts from the identity. Here the
        # predicted residual quaternion is initially zero, so after adding
        # [1,0,0,0] the initial rotation is exactly the identity rotation.
        nn.init.zeros_(self.quaternion_output.weight)
        nn.init.zeros_(self.quaternion_output.bias)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Return one unit quaternion (B,4) for every patch (B,3,k)."""

        features = self.point_mlp(patches)                  # (B, 256, k)
        global_features = torch.max(features, dim=2).values # (B, 256)
        features = self.fc(global_features)                 # (B, 64)
        quaternion = self.quaternion_output(features)       # (B, 4)

        identity = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=quaternion.device,
            dtype=quaternion.dtype,
        )
        quaternion = quaternion + identity.unsqueeze(0)
        return F.normalize(quaternion, p=2, dim=1)


class MainPointNet(nn.Module):
    """Shared point MLP + max pooling + displacement regression."""

    def __init__(self):
        super().__init__()

        self.point_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.displacement_output = nn.Linear(64, 3)

        # Keep PyTorch's normal random weight initialization. Only the bias is
        # zero. Zeroing the weights themselves made the previous experiment
        # converge too easily to the trivial zero-displacement solution.
        nn.init.zeros_(self.displacement_output.bias)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        point_features = self.point_mlp(points)                    # (B, 256, N)
        global_feature = torch.max(point_features, dim=2).values   # (B, 256)
        features = self.fc(global_feature)                          # (B, 64)
        return self.displacement_output(features)                   # (B, 3)


class PointNetDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.quaternion_net = QuaternionInputTNet()
        self.main_net = MainPointNet()

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # (B, N, 3) -> (B, 3, N) for Conv1d input.
        points = patches.transpose(1, 2)

        # 1. Estimate a proper rotation from a unit quaternion.
        quaternion = self.quaternion_net(points)                # (B, 4)
        rotation = quaternion_to_rotation_matrix(quaternion)    # (B, 3, 3)

        # 2. Rotate the local patch to the learned aligned coordinate system.
        aligned_points = torch.bmm(rotation, points)            # (B, 3, N)

        # 3. Predict the normalized inverse-noise displacement there.
        aligned_displacement = self.main_net(aligned_points)    # (B, 3)

        # 4. Bring the vector back to the original coordinate system.
        inverse_rotation = rotation.transpose(1, 2)
        return torch.bmm(
            inverse_rotation,
            aligned_displacement.unsqueeze(2),
        ).squeeze(2)
