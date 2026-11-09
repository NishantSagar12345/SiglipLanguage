"""Evaluate language-conditioned gridworld navigation under SigLIP visual-token reduction.

Run (Kaggle T4 or any CUDA machine):

    python src/evaluate.py --out .                 # writes results/ and figures/
    python src/evaluate.py --out . --fill-readme   # render the README results section

Order of work:
  1. Generate the deterministic episode set (default 300, seed 42) -> results/episodes.json.
  2. Oracle rollout (ground-truth target): must give 100% success at optimal length,
     otherwise the environment/planner is broken and the run stops.
  3. Load SiglipModel in FP32 and VALIDATE: custom 196-token crop scoring must match
     SiglipModel(input_ids, pixel_values).logits_per_image. Stops on failure, before
     any reduced-token configuration is evaluated.
  4. Cast the image tower to FP16 (benchmark precision; logits stay FP32) and run:
     SigLIP 196/144/100/49 tokens, SigLIP-196 with shuffled instructions (language
     control), random candidate, nearest candidate (text-agnostic baselines).
  5. Latency: per-decision GPU perception (single agent; and 64 agents batched),
     encoder only, CPU render+proposal+crop+preprocess, BFS planning.
  6. Metrics with paired bootstrap CIs vs SigLIP-196; CSV/JSON; figures.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import siglip_scorer as S  # noqa: E402
from agent import (NearestPolicy, OraclePolicy, RandomPolicy, SiglipPolicy,  # noqa: E402
                   episode_records, run_policy, trajectory_records)
from gridworld import Renderer, bfs, blocked_cells, generate_episodes, save_episodes  # noqa: E402
from perception import crop_cells, propose_candidates  # noqa: E402

TOKEN_CONFIGS = ["196", "144", "100", "49"]


# ----------------------------------------------------------------------------- environment
def collect_environment(device: torch.device) -> dict:
    import transformers
    env = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "python": platform.python_version(), "platform": platform.platform(),
           "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
           "cuda_available": torch.cuda.is_available(), "torch_cuda_build": torch.version.cuda,
           "device": str(device)}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        env.update({"gpu_name": p.name, "gpu_total_memory_mib": p.total_memory / 2 ** 20,
                    "gpu_compute_capability": f"{p.major}.{p.minor}"})
        try:
            q = "name,driver_version,clocks.sm,clocks.max.sm,power.limit,temperature.gpu"
            env["nvidia_smi"] = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv"],
                                               capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as e:
            env["nvidia_smi"] = f"unavailable: {e}"
    return env


# ----------------------------------------------------------------------------- helpers
def shuffled_text_map(episodes) -> dict[int, str]:
    """Each episode gets the instruction of the next episode (cyclic) whose target differs.

    Deterministic. The shuffled instruction may name a distractor present in the scene;
    it never names this episode's target, so success requires ignoring the language.
    """
    n, out = len(episodes), {}
    for i, ep in enumerate(episodes):
        for k in range(1, n):
            other = episodes[(i + k) % n]
            if other.target.name != ep.target.name:
                out[ep.episode_id] = other.instruction
                break
    return out


def step0_crops(episodes, renderer, margin):
    """(crops, texts, owner) of the first observation of each episode."""
    crops, texts, owner = [], [], []
    for k, ep in enumerate(episodes):
        obs = renderer.render(ep, ep.start)
        c = propose_candidates(obs, ep.grid_size)
        crops += crop_cells(obs, c, ep.grid_size, margin)
        texts += [ep.instruction] * len(c)
        owner += [k] * len(c)
    return crops, texts, np.asarray(owner)


def per_episode_argmax(logits: np.ndarray, owner: np.ndarray) -> np.ndarray:
    return np.asarray([int(np.argmax(logits[owner == k])) for k in np.unique(owner)])


def paired_bootstrap(values: dict[str, np.ndarray], baseline: str, n_boot: int, seed: int) -> dict:
    """Percentile 95% CIs of each mean and of (mean - baseline mean), paired over episodes."""
    keys = list(values)
    V = np.stack([values[k].astype(np.float64) for k in keys])
    n = V.shape[1]
    rng = np.random.default_rng(seed)
    boots = []
    for s in range(0, n_boot, 500):
        idx = rng.integers(0, n, size=(min(500, n_boot - s), n))
        boots.append(V[:, idx].mean(-1))
    Bm = np.concatenate(boots, axis=1)
    b0 = Bm[keys.index(baseline)]
    out = {}
    for i, k in enumerate(keys):
        d = Bm[i] - b0
        out[k] = {"mean": float(V[i].mean()), "ci_low": float(np.percentile(Bm[i], 2.5)),
                  "ci_high": float(np.percentile(Bm[i], 97.5)),
                  "delta": float(V[i].mean() - V[keys.index(baseline)].mean()),
                  "delta_ci_low": float(np.percentile(d, 2.5)), "delta_ci_high": float(np.percentile(d, 97.5))}
    return out


# ----------------------------------------------------------------------------- latency
def measure_latency(scorer, episodes, renderer, margin, warmup, reps, n_states, fleet) -> list[dict]:
    """Interleaved timing over configs; each rep uses one of ``n_states`` step-0 decision states."""
    device = scorer.device
    n_states, fleet = min(n_states, len(episodes)), min(fleet, len(episodes))
    states = episodes[:n_states]
    singles = []
    for ep in states:
        obs = renderer.render(ep, ep.start)
        c = propose_candidates(obs, ep.grid_size)
        px = scorer.preprocess(crop_cells(obs, c, ep.grid_size, margin)).to(device, scorer.dtype)
        singles.append((px, scorer.text_matrix([ep.instruction])))
    fc, ft, fo = step0_crops(episodes[:fleet], renderer, margin)
    fleet_px = scorer.preprocess(fc).to(device, scorer.dtype)
    fleet_t = scorer.text_matrix(ft)
    fleet_owner = torch.as_tensor(fo, device=device)
    w = scorer.wrapper

    def decision(px, t, idx):              # GPU: pixels -> embed -> select -> encoder -> head -> logits -> argmax
        _, pooled = w(px, idx)
        return S.score(pooled, t, scorer.logit_scale, scorer.logit_bias).argmax(0)

    def fleet_decision(idx):               # 64 agents at once: pair logits, per-agent argmax via scatter
        _, pooled = w(fleet_px, idx)
        img = pooled.float() / pooled.float().norm(dim=-1, keepdim=True)
        lg = (img * fleet_t).sum(-1) * scorer.logit_scale + scorer.logit_bias
        best = torch.full((fleet,), -1e9, device=device).scatter_reduce(0, fleet_owner, lg, "amax")
        return best

    rows = []
    with torch.inference_mode():
        tok = {cfg: [w.select(w.embed(px), scorer.configs[cfg]).contiguous() for px, _ in singles]
               for cfg in TOKEN_CONFIGS}
        runners = {}
        for cfg in TOKEN_CONFIGS:
            idx = scorer.configs[cfg]
            runners[(cfg, "decision_gpu_single_agent")] = lambda r, idx=idx: decision(*singles[r % n_states], idx)
            runners[(cfg, "encoder_only_single_agent")] = lambda r, cfg=cfg: w.encode(tok[cfg][r % n_states])
            runners[(cfg, f"decision_gpu_{fleet}_agents_batched")] = lambda r, idx=idx: fleet_decision(idx)
        keys = list(runners)
        for k in keys:
            for r in range(warmup):
                runners[k](r)
        S.sync(device)
        for r in range(reps):
            for k in keys[r % len(keys):] + keys[:r % len(keys)]:
                S.sync(device)
                t0 = time.perf_counter()
                runners[k](r)
                S.sync(device)
                rows.append({"num_tokens": int(k[0]), "mode": k[1], "rep": r,
                             "latency_ms": (time.perf_counter() - t0) * 1e3})
    # config-independent CPU stages
    for r in range(reps):
        ep = states[r % n_states]
        t0 = time.perf_counter()
        obs = renderer.render(ep, ep.start)
        c = propose_candidates(obs, ep.grid_size)
        t1 = time.perf_counter()
        px = scorer.preprocess(crop_cells(obs, c, ep.grid_size, margin))
        t2 = time.perf_counter()
        bfs(ep.grid_size, blocked_cells(ep, ep.target.cell), ep.start, ep.target.cell)
        t3 = time.perf_counter()
        for mode, dt in [("cpu_render_and_proposals", t1 - t0), ("cpu_crop_and_preprocess", t2 - t1),
                         ("cpu_bfs_planning", t3 - t2)]:
            rows.append({"num_tokens": -1, "mode": mode, "rep": r, "latency_ms": dt * 1e3})
    del px
    return rows


def aggregate_latency(df):
    g = df.groupby(["mode", "num_tokens"])["latency_ms"]
    out = g.agg(n="count", median_ms="median", mean_ms="mean", std_ms="std",
                p90_ms=lambda s: float(np.percentile(s, 90))).reset_index()
    base = out[out.num_tokens == 196].set_index("mode")["median_ms"]
    out["speedup_vs_196"] = [base[m] / v if m in base.index else float("nan")
                             for m, v in zip(out["mode"], out.median_ms)]
    return out


# ----------------------------------------------------------------------------- summary
def summarise(df, n_boot: int, seed: int) -> "pd.DataFrame":
    import pandas as pd
    base = "SigLIP 196 tokens"
    pols = list(dict.fromkeys(df.policy))
    by = {p: df[df.policy == p].sort_values("episode_id") for p in pols}
    ids = by[base].episode_id.values
    for p in pols:
        assert (by[p].episode_id.values == ids).all(), "policies must share the same episodes"
    succ = paired_bootstrap({p: by[p].success.values for p in pols}, base, n_boot, seed)
    sel = paired_bootstrap({p: by[p].initial_selection_correct.values for p in pols}, base, n_boot, seed)
    rows = []
    for p in pols:
        d = by[p]
        ok = d[d.success == 1]
        r = {"policy": p, "num_tokens": int(d.num_tokens.iloc[0]), "n_episodes": len(d),
             "success_rate": succ[p]["mean"], "success_ci_low": succ[p]["ci_low"], "success_ci_high": succ[p]["ci_high"],
             "success_delta_vs_196": succ[p]["delta"], "success_delta_ci_low": succ[p]["delta_ci_low"],
             "success_delta_ci_high": succ[p]["delta_ci_high"],
             "target_selection_accuracy_step0": sel[p]["mean"], "selection_ci_low": sel[p]["ci_low"],
             "selection_ci_high": sel[p]["ci_high"],
             "selection_delta_vs_196": sel[p]["delta"], "selection_delta_ci_low": sel[p]["delta_ci_low"],
             "selection_delta_ci_high": sel[p]["delta_ci_high"],
             "per_step_selection_accuracy": float(d.step_selection_accuracy.mean()),
             "mean_steps_successful": float(ok.steps.mean()) if len(ok) else float("nan"),
             "mean_optimal_steps_successful": float(ok.optimal_steps.mean()) if len(ok) else float("nan"),
             "spl": float(d.spl.mean()),
             "mean_selection_switches": float(d.selection_switches.mean()),
             "episodes_with_no_path_events": int((d.no_path_events > 0).sum()),
             "proposal_ok_rate": float(d.proposal_ok_step0.mean())}
        for o, c in d.outcome.value_counts().items():
            r[f"outcome:{o}"] = int(c)
        rows.append(r)
    return pd.DataFrame(rows).fillna({c: 0 for c in pd.DataFrame(rows).columns if c.startswith("outcome:")})


# ----------------------------------------------------------------------------- README
def results_markdown(results_dir: Path) -> str:
    import pandas as pd
    p = json.loads((results_dir / "params.json").read_text())
    env = json.loads((results_dir / "environment.json").read_text())
    v = json.loads((results_dir / "validation.json").read_text())
    s = pd.read_csv(results_dir / "summary.csv")
    lat = pd.read_csv(results_dir / "latency_summary.csv")
    vg = v["siglip_logits_fp32"]
    lines = [f"Measured on **{env.get('gpu_name') or env['device']}**, PyTorch {env['torch']}, Transformers "
             f"{env['transformers']}, image tower **{p['eval_dtype']}** (logits FP32), {p['n_episodes']} episodes "
             f"(seed {p['seed']}), run {env['timestamp_utc']}.", "",
             f"Validation: oracle success {v['oracle']['success_rate']:.0%} with optimal path length in "
             f"{v['oracle']['optimal_length_rate']:.0%} of episodes; pixel proposals equal the true object cells "
             f"in {v['proposals']['match_rate']:.1%} of first observations; 196-token crop scoring vs "
             f"`SiglipModel(...).logits_per_image` (FP32, {vg['n_images']} crops × {vg['n_texts']} instructions): "
             f"max |Δ logit| = {vg['max_abs_diff_logits']:.2e} (|logit| ≤ {vg['max_abs_ref_logit']:.1f}), argmax "
             f"agreement {vg['argmax_agreement']:.0%} — {'**passed**' if vg['passed'] else '**FAILED**'}."]
    if "fp16_vs_fp32_selection" in v:
        f = v["fp16_vs_fp32_selection"]
        lines.append(f"{p['eval_dtype']} vs FP32 first-step target choice agreement: {f['agreement']:.1%} "
                     f"({f['n_episodes']} episodes).")
    lines += ["", "| Policy | Success | 95% CI | Δ vs 196 (pp) [paired 95% CI] | Target selection (step 0) | Δ sel. vs 196 (pp) | "
                  "Steps (successes) | SPL |",
              "|---|---:|---|---|---:|---:|---:|---:|"]
    for r in s.itertuples():
        lines.append(f"| {r.policy} | {r.success_rate * 100:.1f}% | [{r.success_ci_low * 100:.1f}, "
                     f"{r.success_ci_high * 100:.1f}] | {r.success_delta_vs_196 * 100:+.1f} "
                     f"[{r.success_delta_ci_low * 100:+.1f}, {r.success_delta_ci_high * 100:+.1f}] | "
                     f"{r.target_selection_accuracy_step0 * 100:.1f}% | {r.selection_delta_vs_196 * 100:+.1f} | "
                     f"{r.mean_steps_successful:.2f} (opt. {r.mean_optimal_steps_successful:.2f}) | {r.spl:.3f} |")
    oc = [c for c in s.columns if c.startswith("outcome:")]
    lines += ["", "**Episode outcomes (counts)**", "",
              "| Policy | " + " | ".join(c.split(":", 1)[1] for c in oc) + " |",
              "|---|" + "---:|" * len(oc)]
    for _, r in s.iterrows():
        lines.append(f"| {r.policy} | " + " | ".join(str(int(r[c])) for c in oc) + " |")
    lines += ["", "**Latency per decision** (median / p90 ms over timed reps; GPU timings start from preprocessed "
                  "crops already on the GPU, CPU stages are timed separately and do not depend on the token count)", "",
              "| Stage | Tokens | Median (ms) | p90 (ms) | Speed-up vs 196 |", "|---|---:|---:|---:|---:|"]
    for r in lat.sort_values(["mode", "num_tokens"], ascending=[True, False]).itertuples():
        tk = "–" if r.num_tokens < 0 else str(r.num_tokens)
        su = "–" if np.isnan(r.speedup_vs_196) else f"{r.speedup_vs_196:.2f}×"
        lines.append(f"| `{r.mode}` | {tk} | {r.median_ms:.2f} | {r.p90_ms:.2f} | {su} |")
    lines += ["", f"Bootstrap: {p['n_boot']} paired resamples of the episodes, seed {p['seed']}, percentile intervals."]
    return "\n".join(lines) + "\n"


def fill_readme(readme: Path, results_dir: Path) -> None:
    start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
    text = readme.read_text()
    assert start in text and end in text, "README results markers missing"
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    readme.write_text(f"{head}{start}\n{results_markdown(results_dir)}{end}{tail}")
    print(f"README results section updated from {results_dir}")


# ----------------------------------------------------------------------------- main
def main(args) -> None:
    import pandas as pd
    import visualize

    out = Path(args.out)
    results_dir, figures_dir = out / "results", out / "figures"
    results_dir.mkdir(parents=True, exist_ok=True); figures_dir.mkdir(parents=True, exist_ok=True)
    S.set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available - latency results will not be meaningful.")
    env = collect_environment(device)
    print(json.dumps({k: v for k, v in env.items() if k != "nvidia_smi"}, indent=2))
    validation: dict = {}

    # 1. episodes ------------------------------------------------------------------------
    episodes = generate_episodes(args.n_episodes, args.seed, grid_size=args.grid_size, n_objects=args.n_objects,
                                 wall_fraction=args.wall_fraction, min_start_distance=args.min_start_distance)
    save_episodes(episodes, results_dir / "episodes.json")
    renderer = Renderer(cell_px=args.cell_px)
    print(f"{len(episodes)} episodes; optimal steps mean {np.mean([e.optimal_steps for e in episodes]):.2f}")

    # 2. oracle sanity check -----------------------------------------------------------------
    t0 = time.perf_counter()
    oracle = OraclePolicy()
    ost = run_policy(episodes, renderer, oracle, args.max_steps, log_every=0)
    validation["oracle"] = {"success_rate": float(np.mean([s.outcome == "success" for s in ost])),
                            "optimal_length_rate": float(np.mean([s.steps == s.ep.optimal_steps for s in ost])),
                            "seconds": time.perf_counter() - t0}
    validation["proposals"] = {"match_rate": float(np.mean([s.proposal_ok_step0 for s in ost]))}
    print("Oracle / proposal check:", validation["oracle"], validation["proposals"])
    if validation["oracle"]["success_rate"] < 1.0 or validation["oracle"]["optimal_length_rate"] < 1.0:
        (results_dir / "validation.json").write_text(json.dumps(validation, indent=2))
        raise RuntimeError("Oracle policy failed: environment or planner is inconsistent; stopping.")

    # 3. SigLIP FP32 validation gate -----------------------------------------------------
    model = S.load_siglip(args.model_id, device, args.attn_impl)
    scorer = S.SiglipScorer(model, S.load_processor(args.model_id), S.load_tokenizer(args.model_id), device,
                            torch.float32)
    shuffled = shuffled_text_map(episodes)
    scorer.cache_texts([e.instruction for e in episodes] + list(shuffled.values()))   # cached once, FP32
    vc, vt, vo = step0_crops(episodes[:args.val_episodes], renderer, args.crop_margin)
    vpx = scorer.preprocess(vc)
    validation["siglip_logits_fp32"] = S.validate_against_siglip(scorer, vpx, vt, args.atol, args.rtol)
    (results_dir / "validation.json").write_text(json.dumps(validation, indent=2))
    print("FP32 SigLIP logit validation:", json.dumps(validation["siglip_logits_fp32"], indent=2))
    if not validation["siglip_logits_fp32"]["passed"]:
        raise RuntimeError("196-token crop scoring does NOT reproduce SiglipModel logits_per_image within "
                           "tolerance; stopping before evaluating any token configuration. See validation.json")
    sel32 = per_episode_argmax(scorer.pair_logits(vpx, vt, "196").cpu().numpy(), vo)

    # 4. benchmark precision -----------------------------------------------------------------
    eval_dtype = S.DTYPES[args.dtype]
    model.to(eval_dtype)
    scorer.dtype = eval_dtype
    sel16 = per_episode_argmax(scorer.pair_logits(vpx, vt, "196").cpu().numpy(), vo)
    validation["fp16_vs_fp32_selection"] = {"n_episodes": len(sel32), "agreement": float((sel16 == sel32).mean())}
    (results_dir / "validation.json").write_text(json.dumps(validation, indent=2))
    del vpx; gc.collect()

    policies = [SiglipPolicy(scorer, cfg, args.crop_margin, batch=args.batch_size) for cfg in TOKEN_CONFIGS]
    policies += [SiglipPolicy(scorer, "196", args.crop_margin, text_for=lambda ep: shuffled[ep.episode_id],
                              name="SigLIP 196, shuffled instruction", batch=args.batch_size),
                 RandomPolicy(args.seed), NearestPolicy(), oracle]
    tokens_of = {p.name: (int(p.config) if isinstance(p, SiglipPolicy) else -1) for p in policies}
    rows, traj, timing = [], {}, {}
    for pol in policies:
        print(f"Running {pol.name} ...")
        t0 = time.perf_counter()
        states = ost if pol is oracle else run_policy(episodes, renderer, pol, args.max_steps)
        timing[pol.name] = time.perf_counter() - t0
        rows += episode_records(states, pol, {"num_tokens": tokens_of[pol.name],
                                              "uses_language": pol.uses_language,
                                              "uses_ground_truth_target": pol.uses_ground_truth_target})
        traj[pol.name] = trajectory_records(states, pol)
        print(f"  success {np.mean([s.outcome == 'success' for s in states]):.3f} ({timing[pol.name]:.1f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "episodes_results.csv", index=False)
    (results_dir / "trajectories.json").write_text(json.dumps(traj))
    summ = summarise(df, args.n_boot, args.seed)
    summ.to_csv(results_dir / "summary.csv", index=False)
    attr = (df[df.num_tokens > 0].groupby(["policy", "num_tokens", "target_shape"]).initial_selection_correct.mean()
            .rename("selection_accuracy").reset_index())
    attr2 = (df[df.num_tokens > 0].groupby(["policy", "num_tokens", "target_colour"]).initial_selection_correct.mean()
             .rename("selection_accuracy").reset_index())
    pd.concat([attr.rename(columns={"target_shape": "attribute_value"}).assign(attribute="shape"),
               attr2.rename(columns={"target_colour": "attribute_value"}).assign(attribute="colour")]) \
        .to_csv(results_dir / "selection_by_attribute.csv", index=False)

    # 5. latency -----------------------------------------------------------------------------
    lat = pd.DataFrame(measure_latency(scorer, episodes, renderer, args.crop_margin, args.warmup, args.reps,
                                       args.latency_states, args.fleet))
    lat.to_csv(results_dir / "latency_raw.csv", index=False)
    aggregate_latency(lat).to_csv(results_dir / "latency_summary.csv", index=False)

    params = {**{k: v for k, v in vars(args).items() if k not in {"fill_readme"}},
              "eval_dtype": args.dtype, "logits_dtype": "float32",
              "token_configs": {k: int(v.numel()) for k, v in scorer.configs.items()},
              "text_max_length": S.TEXT_MAX_LENGTH, "logit_scale_exp": scorer.logit_scale,
              "logit_bias": scorer.logit_bias, "policy_wall_seconds": timing,
              "distinct_texts_cached": len(scorer._text_cache)}
    (results_dir / "params.json").write_text(json.dumps(params, indent=2))
    (results_dir / "environment.json").write_text(json.dumps(env, indent=2))
    (results_dir / "validation.json").write_text(json.dumps(validation, indent=2))
    del model, scorer; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    figs = visualize.make_all(results_dir, figures_dir, renderer, episodes, crop_margin=args.crop_margin)
    md = results_markdown(results_dir)
    (results_dir / "results_table.md").write_text(md)
    print("\n" + md)
    print("Figures:", *figs, sep="\n  ")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=".")
    p.add_argument("--model-id", default=S.MODEL_ID)
    p.add_argument("--n-episodes", type=int, default=300)
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--n-objects", type=int, default=5)
    p.add_argument("--wall-fraction", type=float, default=0.15)
    p.add_argument("--min-start-distance", type=int, default=3)
    p.add_argument("--cell-px", type=int, default=32)
    p.add_argument("--crop-margin", type=float, default=0.25, help="crop = object cell + margin*cell on each side")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--dtype", default="float16", choices=list(S.DTYPES))
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--batch-size", type=int, default=256, help="crops per GPU batch during rollouts")
    p.add_argument("--val-episodes", type=int, default=8)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--reps", type=int, default=60)
    p.add_argument("--latency-states", type=int, default=20)
    p.add_argument("--fleet", type=int, default=64, help="agents scored together in the batched latency mode")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fill-readme", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    if a.fill_readme:
        fill_readme(Path(a.out) / "README.md", Path(a.out) / "results")
    else:
        main(a)
