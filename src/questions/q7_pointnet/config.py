"""Configuration for Question 7: PointNet mesh denoising."""

from pathlib import Path


# Mesh-disjoint split: test meshes are never used during training/validation.
TRAIN_MODELS = ("BUNNY", "TEAPOT", "SUZANNE", "DOLPHIN", "HAND")
VALIDATION_MODELS = ("DRAGON", "UNICORN")
TEST_MODELS = ("ARMADILLO", "FLASHLIGHT")

# Q7 is trained on all noise families implemented in Q1.
NOISE_TYPES = ("GAUSSIAN", "NORMAL", "UNIFORM", "IMPULSE", "QUANTIZATION")

# 0.0 is intentionally NOT here. A clean variant is added separately in data.py,
# so it never passes through add_noise() (important for quantization).
NOISE_LEVELS = (0.02, 0.05, 0.10, 0.15, 0.20)
ROTATION_AUGMENTATION_PROBABILITY = 0.75

PATCH_SIZE = 64
TRAIN_SAMPLES_PER_VARIANT = 768
VALIDATION_SAMPLES_PER_VARIANT = 1024

BATCH_SIZE = 128
EPOCHS = 60
LEARNING_RATE = 5e-4
SEED = 7

# One decay around the time our previous runs begin to settle.
SCHEDULER_STEP_SIZE = 10 # Decay the learning rate every 10 epochs.
SCHEDULER_GAMMA = 0.5 # Decay the learning rate by a factor of 0.5.

# Stop if validation loss has not improved for this many epochs.
EARLY_STOPPING_PATIENCE = 15

# None -> fresh training, "latest" or "best" -> resume from that checkpoint.
RESUME_FROM = None

# The best validation checkpoint is the natural choice for inference/evaluation.
INFERENCE_CHECKPOINT = "best"

OUTPUT_DIR = Path("outputs") / "pointnet_final"

# Fixed settings used only for the final Q7 comparison with Q4/Q5.
EVALUATION_STRENGTH = 0.30
EVALUATION_ITERATIONS = 10
