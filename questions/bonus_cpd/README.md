# Bonus: correspondence-free dynamic mesh denoising

This module uses non-rigid Coherent Point Drift to restore a common topology
to an independently remeshed dynamic sequence.

Select the dynamic mesh once in the root `config.py`. The prepared sequences
are `"bouncing"` and `"swing"`:

```python
DYNAMIC_MESH_NAME = "swing"
```

Question 8 then uses `resources/dynamic/<name>`. The Bonus uses
`resources/dynamic/<name>_nocorr/clean_remeshed`, and stores the CPD result in
`outputs/cpd_registered_sequence_<name>`. This keeps the cached Bouncing and
Swing results separate.

If the no-correspondence input has not been created yet, run:

```text
python remeshing.py
```

Run the complete Bonus with:

```text
python bonus.py
```

The application first displays the raw remeshed sequence. Press `R` to load
the full CPD-registered sequence, `1` to add noise, and `8` to run temporal
denoising. Press `B` to return to the raw remeshed input. If the registered
sequence does not exist yet, `bonus.py` computes it once and saves it.

The separate two-frame visualization remains available with:

```text
python cpd_demo.py --1 --35
```

The first number is the moving source frame. The second is the fixed target
frame. The forms `python cpd_demo.py 1 35` and
`python cpd_demo.py --1, --35` are also accepted.

In the two-frame demo, the target remains red. The source is blue and moves through the stored CPD
iterations. Its original position remains visible in transparent gray.
The comparison is visualized only with the common-scale point-to-surface error heatmap.

Controls:

- `SPACE`: play or pause the CPD iterations.
- `LEFT` / `RIGHT`: move one iteration.
- `HOME` or `R`: return to iteration zero.
- `END`: show the final registration.
- `E`: switch between blue points and the error heatmap.
- `?`: show the help text.

The CPD mathematical parameters remain in `questions/bonus_cpd/config.py`.
Dataset paths and default frame numbers remain in the root `config.py`.
