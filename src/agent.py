"""Policies (target selection) and the closed-loop, batched rollout.

Perception -> target selection -> planning -> action, repeated every step:

  1. render the RGB observation (agent drawn in its current cell);
  2. propose candidate cells from pixels only (``perception.propose_candidates``);
  3. the policy selects one candidate as the goal;
  4. BFS shortest path to that goal over walls + other objects (ground-truth
     occupancy, allowed for collision-free planning); take the first move;
  5. arriving on the selected cell = STOP. Success = stopped on the true target.

Ground-truth target identity is used ONLY by ``OraclePolicy`` (a labelled
sanity check) and afterwards for scoring. ``SiglipPolicy`` sees the crops and
the instruction string; ``RandomPolicy`` and ``NearestPolicy`` ignore language.

All episodes advance in lockstep so that each step's crops from every active
episode are scored by SigLIP in large GPU batches. Batching does not change
results: every crop is scored independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gridworld import Episode, Renderer, bfs, blocked_cells
from perception import crop_cells, propose_candidates


@dataclass
class AgentState:
    ep: Episode
    pos: tuple[int, int]
    steps: int = 0
    done: bool = False
    stopped_at: tuple | None = None
    outcome: str = ""
    committed: tuple | None = None
    path: list = field(default_factory=list)
    selections: list = field(default_factory=list)
    no_path_events: int = 0
    proposal_ok_step0: bool | None = None
    step0_scores: dict | None = None


class Policy:
    name = "policy"
    uses_language = False
    uses_ground_truth_target = False

    def select(self, states: list[AgentState], obs: list[np.ndarray], cands: list[list[tuple]]) -> list[tuple | None]:
        raise NotImplementedError


class OraclePolicy(Policy):
    """Uses the ground-truth target cell. Sanity check of environment + planner only."""
    name = "oracle (GT target)"
    uses_ground_truth_target = True

    def select(self, states, obs, cands):
        return [s.ep.target.cell for s in states]


class RandomPolicy(Policy):
    """Uniformly random candidate at step 0 (seeded per episode), then committed. Text-agnostic."""
    name = "random candidate"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def select(self, states, obs, cands):
        out = []
        for s, c in zip(states, cands):
            if s.committed is None or s.committed not in c:
                if not c:
                    out.append(None); continue
                rng = np.random.default_rng(self.seed * 100003 + s.ep.episode_id)
                s.committed = c[int(rng.integers(len(c)))]
            out.append(s.committed)
        return out


class NearestPolicy(Policy):
    """Candidate with the shortest collision-free path from the start, then committed. Text-agnostic."""
    name = "nearest candidate"

    def select(self, states, obs, cands):
        out = []
        for s, c in zip(states, cands):
            if s.committed is None or s.committed not in c:
                best, best_d = None, 10 ** 9
                for cell in c:                        # row-major order breaks ties deterministically
                    p = bfs(s.ep.grid_size, blocked_cells(s.ep, cell), s.pos, cell)
                    if p is not None and len(p) < best_d:
                        best, best_d = cell, len(p)
                s.committed = best
            out.append(s.committed)
        return out


class SiglipPolicy(Policy):
    """argmax over candidate crops of the SigLIP logit against the instruction text.

    Re-perceives at every step (no memory, no commitment). ``text_for`` maps an
    episode to the text it is conditioned on: the episode's own instruction by
    default, or another episode's instruction for the shuffled-instruction control.
    """
    uses_language = True

    def __init__(self, scorer, config: str, crop_margin: float = 0.25, text_for=None, name: str | None = None,
                 batch: int = 256):
        self.scorer, self.config, self.margin, self.batch = scorer, config, crop_margin, batch
        self.text_for = text_for or (lambda ep: ep.instruction)
        self.name = name or f"SigLIP {config} tokens"

    def select(self, states, obs, cands):
        crops, texts, owner = [], [], []
        for k, (s, o, c) in enumerate(zip(states, obs, cands)):
            crops += crop_cells(o, c, s.ep.grid_size, self.margin)
            texts += [self.text_for(s.ep)] * len(c)
            owner += [k] * len(c)
        logits = np.empty(0, np.float32)
        if crops:
            chunks = []
            for i in range(0, len(crops), self.batch):          # CPU preprocessing + GPU scoring per chunk
                px = self.scorer.preprocess(crops[i:i + self.batch])
                chunks.append(self.scorer.pair_logits(px, texts[i:i + self.batch], self.config).float().cpu().numpy())
            logits = np.concatenate(chunks)
        owner = np.asarray(owner)
        out = []
        for k, (s, c) in enumerate(zip(states, cands)):
            if not c:
                out.append(None); continue
            lk = logits[owner == k]
            j = int(np.argmax(lk))                              # first max wins on exact ties
            out.append(c[j])
            if s.steps == 0:
                s.step0_scores = {"cells": [list(x) for x in c], "logits": [float(v) for v in lk]}
        return out


def run_policy(episodes: list[Episode], renderer: Renderer, policy: Policy, max_steps: int = 50,
               log_every: int = 10) -> list[AgentState]:
    states = [AgentState(ep=e, pos=e.start, path=[e.start]) for e in episodes]
    step = 0
    while True:
        active = [s for s in states if not s.done]
        if not active:
            break
        obs = [renderer.render(s.ep, s.pos) for s in active]
        cands = [propose_candidates(o, s.ep.grid_size) for o, s in zip(obs, active)]
        for s, c in zip(active, cands):
            if s.steps == 0:
                s.proposal_ok_step0 = sorted(c) == sorted(o.cell for o in s.ep.objects)
        goals = policy.select(active, obs, cands)
        for s, g in zip(active, goals):
            s.selections.append(None if g is None else tuple(g))
            if g is None:
                s.done, s.outcome = True, "no_candidates"
                continue
            p = bfs(s.ep.grid_size, blocked_cells(s.ep, g), s.pos, g)
            if p is None or len(p) < 2:
                s.no_path_events += 1                         # wait in place
            else:
                s.pos = p[1]
            s.steps += 1
            s.path.append(s.pos)
            if s.pos == g:
                s.done, s.stopped_at = True, g                # STOP on the selected object
            elif s.steps >= max_steps:
                s.done, s.outcome = True, "timeout"
        step += 1
        if log_every and step % log_every == 0:
            print(f"    [{policy.name}] step {step}: {sum(not s.done for s in states)} episodes still active")
    for s in states:
        if s.stopped_at is not None:
            s.outcome = classify_stop(s.ep, s.stopped_at)
    return states


def classify_stop(ep: Episode, cell: tuple) -> str:
    """Outcome of stopping on ``cell`` (uses ground truth: evaluation only)."""
    obj = next((o for o in ep.objects if o.cell == tuple(cell)), None)
    t = ep.target
    if obj is None:
        return "stopped_not_on_object"
    if obj.cell == t.cell:
        return "success"
    if obj.colour == t.colour:
        return "wrong_shape (colour right)"
    if obj.shape == t.shape:
        return "wrong_colour (shape right)"
    return "wrong_colour_and_shape"


def object_at(ep: Episode, cell) -> str | None:
    if cell is None:
        return None
    obj = next((o for o in ep.objects if o.cell == tuple(cell)), None)
    return None if obj is None else obj.name


def episode_records(states: list[AgentState], policy: Policy, tag: dict) -> list[dict]:
    rows = []
    for s in states:
        ep, t = s.ep, s.ep.target
        success = s.outcome == "success"
        sel0 = s.selections[0] if s.selections else None
        step_acc = np.mean([x == t.cell for x in s.selections]) if s.selections else 0.0
        switches = sum(1 for a, b in zip(s.selections, s.selections[1:]) if a != b)
        row = {**tag, "policy": policy.name, "episode_id": ep.episode_id, "instruction": ep.instruction,
               "target": t.name, "target_colour": t.colour, "target_shape": t.shape,
               "success": int(success), "outcome": s.outcome, "steps": s.steps,
               "optimal_steps": ep.optimal_steps,
               "spl": float(success) * ep.optimal_steps / max(s.steps, ep.optimal_steps),
               "initial_selection": object_at(ep, sel0),
               "initial_selection_correct": int(sel0 == t.cell),
               "initial_outcome": classify_stop(ep, sel0) if sel0 is not None else "no_candidates",
               "step_selection_accuracy": float(step_acc), "selection_switches": switches,
               "no_path_events": s.no_path_events, "proposal_ok_step0": int(bool(s.proposal_ok_step0))}
        if s.step0_scores:
            lg = np.asarray(s.step0_scores["logits"])
            cells = [tuple(c) for c in s.step0_scores["cells"]]
            order = np.argsort(-lg)
            row["step0_top1_minus_top2_logit"] = float(lg[order[0]] - lg[order[1]]) if len(lg) > 1 else float("nan")
            row["step0_target_rank"] = int(np.where([cells[i] == t.cell for i in order])[0][0]) + 1 \
                if t.cell in cells else -1
        rows.append(row)
    return rows


def trajectory_records(states: list[AgentState], policy: Policy) -> dict:
    return {str(s.ep.episode_id): {"path": [list(p) for p in s.path],
                                   "selections": [None if x is None else list(x) for x in s.selections],
                                   "outcome": s.outcome, "step0_scores": s.step0_scores}
            for s in states}
