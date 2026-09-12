# Question 8 — dynamic meshes

The implementation follows Algorithm 1 of:
G. Arvanitis, A. S. Lalos, K. Moustakas, **Denoising of dynamic 3D meshes via low-rank spectral analysis**, Computers & Graphics 82 (2019), 140–151.

## Input assumption

Place one animation in a folder as ordered OBJ frames, for example:

```text
resources/dynamic/bouncing/
    frame_000.obj
    frame_001.obj
    frame_002.obj
    ...
```

The basic Q8 assumes exactly what the paper assumes: every frame has the same number of vertices, the same triangle connectivity, and therefore direct vertex correspondence. Varying connectivity belongs to the bonus question.

Run:

```bash
python project.py --dynamic-folder resources/dynamic/bouncing
```

## Code map

- `sequence.py`: load frames, verify correspondence, common normalization, dynamic session.
- `spectral.py`: the paper algorithm, step by step in small functions.
- `evaluation.py`: Q2 sequence averages and the requested normal-angle error heatmap.
- `viewer.py`: manual frame navigation and the Q8 controls.
- `config.py`: only the few numerical parameters used by RPCA.

## Paper algorithm -> functions

1. Binary Laplacian `L = D - C` -> `binary_laplacian()`.
2. `L = U Λ U^T` -> `laplacian_eigenbasis()`.
3. `V_hat = U^T V` -> `graph_fourier_transform()`.
4. 0.01% high-frequency energy threshold -> `high_frequency_count()`.
5. Common `k_bar = max(k_bar_i)` -> `common_high_frequency_count()`.
6. Coherent matrices `Ex, Ey, Ez` -> `coherent_matrices()`.
7. Bilateral low-rank RPCA, Eqs. (19)-(21) -> `low_rank_rpca()`.
8. Replace only selected high-frequency coefficients -> `reconstruct_gfts()`.
9. IGFT -> `inverse_graph_fourier_transform()`.

The paper explicitly fixes the 99.99% energy preservation rule. The provided article gives the RPCA update equations but does not state numerical values for rank, soft-threshold lambda, or iteration count. Those three implementation parameters are therefore visible in `config.py` rather than hidden inside the algorithm.
