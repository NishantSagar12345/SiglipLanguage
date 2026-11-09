"""Fast checks of the environment, perception and planner (no model download needed).

Run:  python -m pytest -q tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent import NearestPolicy, OraclePolicy, RandomPolicy, classify_stop, run_policy  # noqa: E402
from gridworld import (MOVES, Episode, Renderer, bfs, blocked_cells, generate_episodes,  # noqa: E402
                       load_episodes, save_episodes)
from perception import crop_cells, propose_candidates  # noqa: E402
from token_selection import build_configs  # noqa: E402

N = 300


@pytest.fixture(scope="module")
def episodes():
    return generate_episodes(N, base_seed=42)


@pytest.fixture(scope="module")
def renderer():
    return Renderer()


def test_generation_is_deterministic(episodes):
    again = generate_episodes(N, base_seed=42)
    assert [e.to_dict() for e in episodes] == [e.to_dict() for e in again]
    assert generate_episodes(5, base_seed=7)[0].to_dict() != episodes[0].to_dict()


def test_json_roundtrip(episodes, tmp_path):
    save_episodes(episodes, tmp_path / "eps.json")
    assert [e.to_dict() for e in load_episodes(tmp_path / "eps.json")] == [e.to_dict() for e in episodes]


def test_scene_constraints(episodes):
    for ep in episodes:
        t = ep.target
        assert ep.instruction.endswith(f"the {t.colour} {t.shape}.")
        assert any(o.colour == t.colour and o.shape != t.shape for o in ep.objects)
        assert any(o.shape == t.shape and o.colour != t.colour for o in ep.objects)
        for o in ep.objects:                        # every object reachable, so any choice is executable
            assert bfs(ep.grid_size, blocked_cells(ep, o.cell), ep.start, o.cell) is not None


def test_no_start_proximity_shortcut(episodes):
    # target is the nearest object in roughly 1/K of episodes, not systematically
    near = np.mean([e.meta["object_distances"][e.target_index] == min(e.meta["object_distances"]) for e in episodes])
    assert 0.1 < near < 0.45, near


def test_bfs_paths_are_valid_and_shortest(episodes):
    for ep in episodes[:50]:
        goal = ep.target.cell
        blocked = blocked_cells(ep, goal)
        p = bfs(ep.grid_size, blocked, ep.start, goal)
        assert p[0] == ep.start and p[-1] == goal and len(p) - 1 == ep.optimal_steps
        for a, b in zip(p, p[1:]):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 and b not in blocked - {goal}


def test_proposals_match_objects(episodes, renderer):
    for ep in episodes:
        cells = sorted(o.cell for o in ep.objects)
        assert sorted(propose_candidates(renderer.render(ep, ep.start), ep.grid_size)) == cells
        for o in ep.objects[:2]:                    # agent sprite next to an object must not add/remove candidates
            for dr, dc in MOVES.values():
                p = (o.cell[0] + dr, o.cell[1] + dc)
                if 0 <= p[0] < ep.grid_size and 0 <= p[1] < ep.grid_size and p not in blocked_cells(ep):
                    assert sorted(propose_candidates(renderer.render(ep, p), ep.grid_size)) == cells


def test_crops(episodes, renderer):
    ep = episodes[0]
    obs = renderer.render(ep, ep.start)
    crops = crop_cells(obs, [o.cell for o in ep.objects], ep.grid_size, margin=0.25)
    assert all(c.shape == (48, 48, 3) and c.dtype == np.uint8 for c in crops)


def test_oracle_is_perfect(episodes, renderer):
    st = run_policy(episodes, renderer, OraclePolicy(), max_steps=50, log_every=0)
    assert all(s.outcome == "success" and s.steps == s.ep.optimal_steps for s in st)


def test_text_agnostic_baselines_near_chance(episodes, renderer):
    k = len(episodes[0].objects)
    for pol in (RandomPolicy(42), NearestPolicy()):
        st = run_policy(episodes, renderer, pol, max_steps=50, log_every=0)
        rate = np.mean([s.outcome == "success" for s in st])
        assert abs(rate - 1 / k) < 0.08, (pol.name, rate)
        again = run_policy(episodes, renderer, pol, max_steps=50, log_every=0)
        assert [s.outcome for s in st] == [s.outcome for s in again]          # deterministic


def test_failure_classification(episodes):
    ep = episodes[0]
    t = ep.target
    for o in ep.objects:
        c = classify_stop(ep, o.cell)
        if o.cell == t.cell:
            assert c == "success"
        elif o.colour == t.colour:
            assert c.startswith("wrong_shape")
        elif o.shape == t.shape:
            assert c.startswith("wrong_colour (")


def test_token_configs_unchanged():
    cfg = build_configs()
    assert {k: v.numel() for k, v in cfg.items()} == {"196": 196, "144": 144, "100": 100, "49": 49}
    assert cfg["49"].tolist()[:7] == [0, 2, 4, 6, 9, 11, 13]      # same indices as efficient-vit-bench
