# Bonus: two-frame CPD demo

This module uses non-rigid Coherent Point Drift to restore a common topology
to an independently remeshed dynamic sequence.

Select the dynamic mesh once in the root `config.py`:

```python
DYNAMIC_MESH_NAME = "bouncing"
```

Question 8 then uses `resources/dynamic/bouncing`. The Bonus uses
`resources/dynamic/bouncing_nocorr/clean_remeshed`.

Run the complete Bonus with:

```text
python bonus.py
```

The application first displays the raw remeshed sequence. Press `R` to load
the full CPD-registered sequence, `1` to add noise, and `8` to run temporal
denoising. Press `B` to return to the raw remeshed input. If the registered
sequence does not exist yet, `bonus.py` computes it once and saves it.

The separate two-frame visualization remains available with:

Run any other pair with:

```text
python cpd_demo.py --1 --35
```

The first number is the moving source frame. The second is the fixed target
frame. The forms `python cpd_demo.py 1 35` and
`python cpd_demo.py --1, --35` are also accepted.

In the two-frame demo, the target remains red. The source is blue and moves through the stored CPD
iterations. Its original position remains visible in transparent gray.
Chamfer and point-to-surface errors are printed in the output panel.

Controls:

- `SPACE`: play or pause the CPD iterations.
- `LEFT` / `RIGHT`: move one iteration.
- `HOME` or `R`: return to iteration zero.
- `END`: show the final registration.
- `E`: switch between blue points and the error heatmap.
- `?`: show the help text.

The CPD mathematical parameters remain in `questions/bonus_cpd/config.py`.
Dataset paths and default frame numbers remain in the root `config.py`.
