"""Language-conditioned 2D gridworld with rendered RGB observations.

Each episode has a GxG grid (default 10x10), wall cells, K objects (default 5)
drawn from 5 colours x 3 shapes, an agent start cell and an instruction such as
"Navigate to the red circle."

Scene constraints (so that both the colour word and the shape word matter):
  * every (colour, shape) pair is unique within a scene;
  * at least one distractor shares the target's colour (different shape) and
    at least one shares its shape (different colour);
  * from the start, every object is reachable without passing through walls or
    other objects (so that whichever object the agent selects can be reached).

Episodes are generated from ``np.random.default_rng(base_seed + episode_id)``
and saved to JSON, so the same scenes, obstacles, starts and instructions are
used for every token configuration and baseline.

Rendering: light-grey floor, faint grid lines, dark-grey walls, saturated
filled shapes, and the agent as a black "X". Shapes are drawn at 4x and
down-sampled (anti-aliasing). The static scene is rendered once per episode;
the agent sprite is composited per step.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COLOURS = {  # all strongly saturated so they are separable from grey floor/walls/agent
    "red": (220, 40, 40),
    "green": (30, 160, 60),
    "blue": (40, 80, 225),
    "yellow": (235, 195, 20),
    "purple": (145, 55, 195),
}
SHAPES = ["circle", "square", "triangle"]
TEMPLATES = ["Navigate to the {c} {s}.", "Go to the {c} {s}.", "Move to the {c} {s}.", "Find the {c} {s}."]

FLOOR = (232, 232, 232)
GRIDLINE = (210, 210, 210)
WALL = (85, 85, 85)
AGENT = (0, 0, 0)
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


@dataclass
class SceneObject:
    cell: tuple[int, int]
    colour: str
    shape: str

    @property
    def name(self) -> str:
        return f"{self.colour} {self.shape}"


@dataclass
class Episode:
    episode_id: int
    seed: int
    grid_size: int
    walls: list[tuple[int, int]]
    objects: list[SceneObject]
    target_index: int
    start: tuple[int, int]
    instruction: str
    template_index: int
    optimal_steps: int = -1
    meta: dict = field(default_factory=dict)

    @property
    def target(self) -> SceneObject:
        return self.objects[self.target_index]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_name"] = self.target.name
        return d

    @staticmethod
    def from_dict(d: dict) -> "Episode":
        d = dict(d)
        d.pop("target_name", None)
        d["walls"] = [tuple(w) for w in d["walls"]]
        d["start"] = tuple(d["start"])
        d["objects"] = [SceneObject(tuple(o["cell"]), o["colour"], o["shape"]) for o in d["objects"]]
        return Episode(**d)


# ----------------------------------------------------------------------------- grid search
def bfs(grid_size: int, blocked: set, start: tuple, goal: tuple) -> list[tuple] | None:
    """Shortest 4-connected path start -> goal (inclusive) avoiding ``blocked``; None if unreachable.

    Neighbour order is fixed (up, down, left, right), so ties are broken deterministically.
    ``goal`` itself may be in ``blocked`` (objects are goals but obstacles to everything else).
    """
    if start == goal:
        return [start]
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        for dr, dc in MOVES.values():
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < grid_size and 0 <= nxt[1] < grid_size) or nxt in prev:
                continue
            if nxt != goal and nxt in blocked:
                continue
            prev[nxt] = cur
            if nxt == goal:
                path = [nxt]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            q.append(nxt)
    return None


def blocked_cells(ep: Episode, goal: tuple | None = None) -> set:
    """Walls plus every object cell except ``goal``."""
    return set(ep.walls) | {o.cell for o in ep.objects if o.cell != goal}


# ----------------------------------------------------------------------------- generation
def generate_episode(episode_id: int, base_seed: int = 42, grid_size: int = 10, n_objects: int = 5,
                     wall_fraction: float = 0.15, min_start_distance: int = 3, max_tries: int = 1000) -> Episode:
    rng = np.random.default_rng(base_seed + episode_id)
    colours, cells_all = list(COLOURS), [(r, c) for r in range(grid_size) for c in range(grid_size)]
    for attempt in range(max_tries):
        # objects: target + same-colour + same-shape distractors + others (all pairs unique)
        tc, ts = colours[rng.integers(len(colours))], SHAPES[rng.integers(len(SHAPES))]
        pairs = [(tc, ts),
                 (tc, rng.choice([s for s in SHAPES if s != ts])),
                 (rng.choice([c for c in colours if c != tc]), ts)]
        rest = [(c, s) for c in colours for s in SHAPES if (c, s) not in pairs]
        for j in rng.permutation(len(rest))[: n_objects - len(pairs)]:
            pairs.append(rest[j])
        pairs = [(str(c), str(s)) for c, s in pairs]
        order = rng.permutation(len(pairs))                 # target position in list is random
        pairs = [pairs[i] for i in order]
        target_index = int(np.where(order == 0)[0][0])

        n_walls = int(round(wall_fraction * grid_size ** 2))
        perm = rng.permutation(len(cells_all))
        chosen = [cells_all[i] for i in perm[: n_walls + n_objects + 1]]
        walls, obj_cells, start = chosen[:n_walls], chosen[n_walls:n_walls + n_objects], chosen[-1]
        objects = [SceneObject(tuple(map(int, cell)), c, s) for cell, (c, s) in zip(obj_cells, pairs)]
        start = tuple(map(int, start))
        walls = [tuple(map(int, w)) for w in walls]
        tmpl = int(rng.integers(len(TEMPLATES)))
        ep = Episode(episode_id, base_seed + episode_id, grid_size, walls, objects, target_index, start,
                     TEMPLATES[tmpl].format(c=tc, s=ts), tmpl)
        paths = [bfs(grid_size, blocked_cells(ep, o.cell), start, o.cell) for o in objects]
        if any(p is None for p in paths):
            continue
        if min(len(p) - 1 for p in paths) < min_start_distance:   # applies to every object, so start
            continue                                              # proximity carries no target information
        ep.optimal_steps = len(paths[target_index]) - 1
        ep.meta = {"generation_attempts": attempt + 1,
                   "object_distances": [len(p) - 1 for p in paths]}
        return ep
    raise RuntimeError(f"could not generate a valid episode {episode_id}")


def generate_episodes(n: int, base_seed: int = 42, **kw) -> list[Episode]:
    eps = [generate_episode(i, base_seed, **kw) for i in range(n)]
    validate_episodes(eps)
    return eps


def validate_episodes(eps: list[Episode]) -> None:
    for ep in eps:
        names = [o.name for o in ep.objects]
        cells = [o.cell for o in ep.objects]
        assert len(set(names)) == len(names), f"duplicate objects in episode {ep.episode_id}"
        assert len(set(cells)) == len(cells) and not set(cells) & set(ep.walls)
        assert ep.start not in set(cells) | set(ep.walls)
        t = ep.target
        assert any(o.colour == t.colour and o.shape != t.shape for o in ep.objects)
        assert any(o.shape == t.shape and o.colour != t.colour for o in ep.objects)
        assert ep.instruction == TEMPLATES[ep.template_index].format(c=t.colour, s=t.shape)
        assert ep.optimal_steps >= 1


def save_episodes(eps: list[Episode], path: Path) -> None:
    Path(path).write_text(json.dumps([e.to_dict() for e in eps], indent=1))


def load_episodes(path: Path) -> list[Episode]:
    return [Episode.from_dict(d) for d in json.loads(Path(path).read_text())]


# ----------------------------------------------------------------------------- rendering
class Renderer:
    """Renders episodes to uint8 RGB arrays of shape (G*cell_px, G*cell_px, 3)."""

    def __init__(self, cell_px: int = 32, supersample: int = 4):
        self.cell_px, self.ss = cell_px, supersample
        self._static: dict[int, np.ndarray] = {}
        self._agent = self._agent_sprite()

    def _shape(self, draw: ImageDraw.ImageDraw, r: int, c: int, shape: str, rgb, cp: int, pad: float = 0.16):
        x0, y0 = c * cp + pad * cp, r * cp + pad * cp
        x1, y1 = (c + 1) * cp - pad * cp, (r + 1) * cp - pad * cp
        if shape == "circle":
            draw.ellipse([x0, y0, x1, y1], fill=rgb)
        elif shape == "square":
            inset = 0.06 * cp
            draw.rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], fill=rgb)
        elif shape == "triangle":
            draw.polygon([((x0 + x1) / 2, y0), (x1, y1), (x0, y1)], fill=rgb)
        else:
            raise ValueError(shape)

    def static(self, ep: Episode) -> np.ndarray:
        if ep.episode_id in self._static:
            return self._static[ep.episode_id]
        G, cp = ep.grid_size, self.cell_px * self.ss
        img = Image.new("RGB", (G * cp, G * cp), FLOOR)
        d = ImageDraw.Draw(img)
        for i in range(G + 1):
            d.line([(0, i * cp), (G * cp, i * cp)], fill=GRIDLINE, width=self.ss)
            d.line([(i * cp, 0), (i * cp, G * cp)], fill=GRIDLINE, width=self.ss)
        for r, c in ep.walls:
            d.rectangle([c * cp, r * cp, (c + 1) * cp - 1, (r + 1) * cp - 1], fill=WALL)
        for o in ep.objects:
            self._shape(d, o.cell[0], o.cell[1], o.shape, COLOURS[o.colour], cp)
        arr = np.asarray(img.resize((G * self.cell_px,) * 2, Image.LANCZOS), dtype=np.uint8)
        self._static[ep.episode_id] = arr
        return arr

    def _agent_sprite(self) -> np.ndarray:
        cp = self.cell_px * self.ss
        img = Image.new("L", (cp, cp), 0)
        d = ImageDraw.Draw(img)
        a, b, w = 0.22 * cp, 0.78 * cp, int(0.14 * cp)
        d.line([(a, a), (b, b)], fill=255, width=w)
        d.line([(a, b), (b, a)], fill=255, width=w)
        return np.asarray(img.resize((self.cell_px,) * 2, Image.LANCZOS), dtype=np.float32)[..., None] / 255.0

    def render(self, ep: Episode, agent: tuple[int, int]) -> np.ndarray:
        img = self.static(ep).copy()
        cp = self.cell_px
        r, c = agent
        tile = img[r * cp:(r + 1) * cp, c * cp:(c + 1) * cp].astype(np.float32)
        tile = tile * (1 - self._agent) + np.asarray(AGENT, np.float32) * self._agent
        img[r * cp:(r + 1) * cp, c * cp:(c + 1) * cp] = tile.round().astype(np.uint8)
        return img

    def clear_cache(self) -> None:
        self._static.clear()
