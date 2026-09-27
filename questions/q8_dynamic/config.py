# preserve 99.99% of the GFT energy and process the
# high-frequency components containing the remaining 0.01%.
# Equations (8)-(10): process the high-frequency components
# containing 0.01% of the total GFT energy.
HIGH_FREQUENCY_ENERGY_FRACTION = 1e-4

# Tested setting for Bouncing, matching the paper's σ_l = 0.20.
DEFAULT_DYNAMIC_NOISE_LEVEL = 0.20

# Full eigendecomposition settings.
EIGEN_DTYPE = "float32"
EIGH_DRIVER = "evd"

# RPCA parameters.
RPCA_RANK = 5
SHORT_SEQUENCE_RPCA_RANK = 3
SHORT_SEQUENCE_MAX_FRAMES = 12
RPCA_THRESHOLD_FACTOR = 0.25
RPCA_ITERATIONS = 20
RPCA_TOLERANCE = 1e-6
RPCA_SEED = 7

NOISE_SEED = 100

# Use one common heatmap scale for noisy and denoised results.
HEATMAP_PERCENTILE = 99.0
