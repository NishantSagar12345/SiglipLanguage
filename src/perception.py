"""Pixel-only candidate proposals and crops.

The agent never reads object identities from the environment. Candidate object
*locations* are found from the rendered RGB frame alone: floor, grid lines,
walls and the agent are achromatic (grey/black), while every object colour is
strongly saturated, so a cell is a candidate if enough of its pixels have high
chroma (max(R,G,B) - min(R,G,B)). Which candidate matches the instruction is
decided only by SigLIP (or by a baseline policy).

Crops are square windows centred on a candidate cell, ``1 + 2*margin`` cells
wide, padded with the floor colour at the borders, then passed to the SigLIP
image processor (resize to 224x224, normalise).
"""
from __future__ import annotations

import numpy as np

from gridworld import FLOOR


def propose_candidates(obs: np.ndarray, grid_size: int, chroma_threshold: int = 60,
                       min_fraction: float = 0.08) -> list[tuple[int, int]]:
    """Cells whose fraction of saturated pixels exceeds ``min_fraction`` (row-major order)."""
    H, W, _ = obs.shape
    cp = H // grid_size
    assert H == W == cp * grid_size, obs.shape
    o = obs.astype(np.int16)
    chroma = o.max(-1) - o.min(-1)
    frac = (chroma > chroma_threshold).reshape(grid_size, cp, grid_size, cp).mean(axis=(1, 3))
    rr, cc = np.nonzero(frac > min_fraction)
    return [(int(r), int(c)) for r, c in zip(rr, cc)]


def crop_cells(obs: np.ndarray, cells: list[tuple[int, int]], grid_size: int, margin: float = 0.25) -> list[np.ndarray]:
    H = obs.shape[0]
    cp = H // grid_size
    m = int(round(margin * cp))
    pad = np.empty((H + 2 * m, H + 2 * m, 3), np.uint8)
    pad[:] = FLOOR
    pad[m:m + H, m:m + H] = obs
    size = cp + 2 * m
    return [pad[r * cp:r * cp + size, c * cp:c * cp + size].copy() for r, c in cells]
