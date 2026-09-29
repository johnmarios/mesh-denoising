# Mesh Denoising — Project 3

## 3D Computational Geometry and Computer Vision 2025–2026


## Setup

Open a terminal in the project folder:

```bash
cd "C:\path to folder"
```

Activate the Conda environment:

```bash
conda activate "env name"
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

---

## Questions 1–5 and 7

Run the main program by selecting one of the supported models:

```bash
python project.py --model bunny
python project.py --model armadillo
python project.py --model teapot
python project.py --model dragon
```

---

## PointNet Training and Evaluation

Train the PointNet model:

```bash
python project.py --train-pointnet
```

Evaluate the trained PointNet model:

```bash
python project.py --evaluate-pointnet
```

Launch TensorBoard:

```bash
tensorboard --logdir outputs\pointnet_final\tensorboard
```

---

## Question 6

Run the experiments on the reference mesh:

```bash
python q6.py reference
```

Run the experiments on the remeshed mesh:

```bash
python q6.py remeshed
```

Run the experiments on the mesh containing holes:

```bash
python q6.py holes
```

---

## Question 8 — Dynamic Meshes

Run the dynamic-mesh denoising pipeline for the **Bouncing** sequence:

```bash
python q8.py bouncing
```

Run the same pipeline for the **Swing** sequence:

```bash
python q8.py swing
```

---

## Bonus — Dynamic Meshes Without Known Correspondence

### Create Remeshed Sequences

Generate independently remeshed sequences:

```bash
python remeshing.py bouncing
python remeshing.py swing
```

### CPD Between Two Frames

Run Coherent Point Drift between frame 1 and frame 35:

```bash
python cpd_demo.py bouncing --1 --35
python cpd_demo.py swing --1 --35
```

For any other pair of frames:

```bash
python cpd_demo.py bouncing SOURCE_FRAME TARGET_FRAME
python cpd_demo.py swing SOURCE_FRAME TARGET_FRAME
```

Replace `SOURCE_FRAME` and `TARGET_FRAME` with the desired frame indices.

### Full Bonus Pipeline

Run the complete bonus pipeline for the **Bouncing** sequence:

```bash
python bonus.py bouncing
```

Run the complete bonus pipeline for the **Swing** sequence:

```bash
python bonus.py swing
```

---
