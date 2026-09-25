"""Small, explicit configuration for the CPD registration demo."""

# CPD uses only these independently sampled surface points.
SOURCE_CONTROL_POINTS = 350
TARGET_CONTROL_POINTS = 400
CONTROL_CANDIDATE_MULTIPLIER = 8

# These independent samples are used only for the Chamfer evaluation.
EVALUATION_POINTS = 2_000

# Non-rigid CPD parameters.
# A wide kernel and strong regularization keep the body coherent.
# Smaller values can reduce Chamfer by folding the human shape.
BETA = 0.90
REGULARIZATION = 300.0
OUTLIER_WEIGHT = 0.05
MAX_ITERATIONS = 40
TOLERANCE = 1e-5

RANDOM_SEED = 7
SECONDS_PER_ITERATION = 0.35
POINT_SIZE = 0.8

# Errors at or above this percentile are displayed with the reddest color.
# Using one common limit makes the before/after heatmaps comparable.
HEATMAP_LIMIT_PERCENTILE = 95.0

# Full-sequence registration used by bonus.py. Frame 0 is the canonical mesh
# whose topology is transferred independently onto every target frame.
SEQUENCE_CONTROL_POINTS = 300
SEQUENCE_BETA = 0.50
SEQUENCE_REGULARIZATION = 100.0
