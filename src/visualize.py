"""Figures for the SigLIP gridworld navigation experiment (all read from results/)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from perception import crop_cells

# Validated categorical palette (fixed order; checked with a CVD/contrast validator).
# Slots 3-5 are below 3:1 contrast on white, so every chart also has a legend and
# the numbers are in results/*.csv and the README tables.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
TOKENS = [196, 144, 100, 49]
OUTCOMES = ["success", "wrong_shape (colour right)", "wrong_colour (shape right)",
            "wrong_colour_and_shape", "timeout", "no_candidates"]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "text.color": INK, "figure.dpi": 120, "legend.frameon": False, "lines.linewidth": 2})
    return plt


def _save(fig, figures_dir: Path, name: str, paths: list):
    fig.tight_layout(rect=(0, 0, 1, 0.93) if fig._suptitle is not None else (0, 0, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    paths.append(figures_dir / f"{name}.png")
    import matplotlib.pyplot as plt
    plt.close(fig)


def make_all(results_dir: Path, figures_dir: Path, renderer, episodes, crop_margin: float = 0.25) -> list[Path]:
    import pandas as pd
    plt = _style()
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    summ = pd.read_csv(results_dir / "summary.csv")
    df = pd.read_csv(results_dir / "episodes_results.csv")
    lat = pd.read_csv(results_dir / "latency_summary.csv")
    traj = json.loads((results_dir / "trajectories.json").read_text())
    env = json.loads((results_dir / "environment.json").read_text())
    params = json.loads((results_dir / "params.json").read_text())
    gpu, prec, n = env.get("gpu_name") or env["device"], params["eval_dtype"], params["n_episodes"]
    sig = summ[summ.num_tokens > 0].copy()
    main = sig[~sig.policy.str.contains("shuffled")].sort_values("num_tokens", ascending=False)
    base = {r.policy: r for r in summ.itertuples()}
    paths: list[Path] = []
    by_id = {e.episode_id: e for e in episodes}

    # fig1: success + target-selection accuracy vs tokens -----------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    for col, (m, lo, hi, lab, c, mk) in enumerate([
            ("success_rate", "success_ci_low", "success_ci_high", "task success", SLOTS[0], "o"),
            ("target_selection_accuracy_step0", "selection_ci_low", "selection_ci_high",
             "target selection (step 0)", SLOTS[1], "s")]):
        y = main[m] * 100
        ax.errorbar(main.num_tokens, y, yerr=[y - main[lo] * 100, main[hi] * 100 - y], color=c, marker=mk,
                    ms=6, capsize=3, label=lab)
    for name, ls in [("SigLIP 196, shuffled instruction", "--"), ("random candidate", ":"),
                     ("nearest candidate", "-.")]:
        if name in base:
            v = base[name].success_rate * 100
            ax.axhline(v, color=INK2, ls=ls, lw=1.2)
            ax.text(200, v + 1.2, f"{name}: {v:.0f}%", fontsize=7.5, color=INK2, ha="right")
    ax.set_xticks(TOKENS); ax.set_xlim(40, 205); ax.set_ylim(0, 105)
    ax.set_xlabel("Visual tokens per crop (N)"); ax.set_ylabel("Rate over episodes (%)")
    ax.set_title(f"Navigation success and target selection vs visual tokens\n"
                 f"{n} episodes · 95% bootstrap CI · {gpu} · {prec}", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=8.5)
    _save(fig, figures_dir, "fig1_success_vs_tokens", paths)

    # fig2: efficiency vs task success ------------------------------------------------------------------
    modes = [m for m in ["decision_gpu_single_agent"] + sorted(m for m in lat["mode"].unique() if "agents_batched" in m)
             if m in set(lat["mode"])]
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 3.9), sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        d = lat[lat["mode"] == mode].set_index("num_tokens")
        x = [d.loc[t, "median_ms"] for t in main.num_tokens]
        y = main.success_rate.values * 100
        ax.errorbar(x, y, yerr=[y - main.success_ci_low * 100, main.success_ci_high * 100 - y],
                    color=SLOTS[0], marker="o", ms=6, capsize=3)
        for xi, yi, t in zip(x, y, main.num_tokens):
            ax.annotate(f"{t} tokens", (xi, yi), textcoords="offset points", xytext=(6, 5), fontsize=8, color=INK)
        if "random candidate" in base:
            ax.axhline(base["random candidate"].success_rate * 100, color=INK2, ls=":", lw=1.2)
            ax.text(ax.get_xlim()[1], base["random candidate"].success_rate * 100 + 1.5, "random", fontsize=7.5,
                    color=INK2, ha="right")
        who = "1 agent (~5 crops)" if "single" in mode else mode.replace("decision_gpu_", "").replace("_", " ")
        ax.set_xlabel(f"Median GPU perception latency per decision (ms)\n{who}", fontsize=9)
    axes[0][0].set_ylabel("Task success (%)")
    axes[0][0].set_ylim(0, 105)
    fig.suptitle(f"Efficiency vs task success · {gpu} · {prec} · sdpa · preprocessed crops on GPU → logits",
                 fontsize=9.5)
    _save(fig, figures_dir, "fig2_efficiency_vs_success", paths)

    # fig3: failure categories ----------------------------------------------------------------------
    order = list(main.policy) + [p for p in ["SigLIP 196, shuffled instruction", "random candidate",
                                             "nearest candidate", "oracle (GT target)"] if p in base]
    cats = [c for c in OUTCOMES if f"outcome:{c}" in summ.columns]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    left = np.zeros(len(order))
    s_idx = summ.set_index("policy")
    for i, c in enumerate(cats):
        v = np.array([s_idx.loc[p, f"outcome:{c}"] for p in order], float) / n * 100
        ax.barh(range(len(order)), v, left=left, color=SLOTS[i % len(SLOTS)], edgecolor="white", linewidth=2,
                height=0.7, label=c)
        for j, (l, w) in enumerate(zip(left, v)):
            if w >= 7:
                ax.text(l + w / 2, j, f"{w:.0f}", ha="center", va="center", fontsize=7.5, color="white")
        left += v
    ax.set_yticks(range(len(order)), order); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("Share of episodes (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Episode outcomes ({n} episodes; failures classified by the object the agent stopped on)",
                 fontsize=9.5)
    ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.18))
    _save(fig, figures_dir, "fig3_failure_categories", paths)

    # fig4: selection accuracy by target attribute -----------------------------------------------------
    att = pd.read_csv(results_dir / "selection_by_attribute.csv")
    att = att[~att.policy.str.contains("shuffled")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True, gridspec_kw={"width_ratios": [3, 5]})
    for ax, a in zip(axes, ["shape", "colour"]):
        d = att[att.attribute == a]
        vals = sorted(d.attribute_value.unique())
        w = 0.8 / len(TOKENS)
        for i, t in enumerate(TOKENS):
            dd = d[d.num_tokens == t].set_index("attribute_value").reindex(vals)
            ax.bar(np.arange(len(vals)) + (i - 1.5) * w, dd.selection_accuracy * 100, w * 0.92,
                   color=SLOTS[i], label=f"{t} tokens")
        ax.set_xticks(range(len(vals)), vals); ax.grid(axis="x", visible=False)
        ax.set_xlabel(f"Target {a}")
    axes[0].set_ylabel("Target selection accuracy, step 0 (%)"); axes[0].set_ylim(0, 105)
    axes[1].legend(ncol=4, fontsize=8, loc="upper right")
    fig.suptitle("Target selection by target attribute (SigLIP, own instruction)", fontsize=9.5)
    _save(fig, figures_dir, "fig4_selection_by_attribute", paths)

    # fig5: sample episodes with trajectories --------------------------------------------------------
    show = [p for p in ["SigLIP 196 tokens", "SigLIP 100 tokens", "SigLIP 49 tokens"] if p in traj]
    if show:
        piv = df[df.policy.isin(show)].pivot(index="episode_id", columns="policy", values="success")
        picks = []
        rules = [("196 ok, 49 fail", lambda r: r.get(show[0]) == 1 and r.get(show[-1]) == 0),
                 ("all ok", lambda r: all(r.get(p) == 1 for p in show)),
                 ("196 fail", lambda r: r.get(show[0]) == 0)]
        for label, rule in rules:                       # first episodes (by id) matching each rule
            hits = [i for i, r in piv.iterrows() if rule(r.to_dict())][: 2 if label != "196 fail" else 1]
            picks += [(i, label) for i in hits]
        picks = picks[:5] or [(int(piv.index[0]), "first episode")]
        fig, axes = plt.subplots(1, len(picks), figsize=(3.3 * len(picks), 3.9), squeeze=False)
        for ax, (eid, label) in zip(axes[0], picks):
            ep = by_id[int(eid)]
            img = renderer.render(ep, ep.start)
            cp = img.shape[0] / ep.grid_size
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            tr, tc = ep.target.cell
            ax.add_patch(plt.Rectangle((tc * cp, tr * cp), cp, cp, fill=False, ec=INK, lw=2.2, ls="--"))
            for k, p in enumerate(show):
                path = np.array(traj[p][str(eid)]["path"], float)
                off = (k - (len(show) - 1) / 2) * 0.12 * cp
                ax.plot(path[:, 1] * cp + cp / 2 + off, path[:, 0] * cp + cp / 2 + off, color=SLOTS[k], lw=2,
                        alpha=0.95, label=f"{p.replace('SigLIP ', '')}: {traj[p][str(eid)]['outcome'].split(' ')[0]}")
                end = path[-1]
                ax.plot(end[1] * cp + cp / 2 + off, end[0] * cp + cp / 2 + off, "o", color=SLOTS[k], ms=7,
                        mec="white", mew=1.5)
            ax.set_title(f"#{eid} [{label}]\n“{ep.instruction}”", fontsize=8.5)
            ax.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=1)
        fig.suptitle("Sample episodes: start frame and trajectory per token count\n"
                     "(dot = where the agent stopped, dashed box = true target)", fontsize=9.5)
        _save(fig, figures_dir, "fig5_sample_trajectories", paths)

        # fig6: step-0 crops with SigLIP logits per token count for one episode
        eid = int(picks[0][0])
        ep = by_id[eid]
        s0 = {p: traj[p][str(eid)]["step0_scores"] for p in show}
        cells = [tuple(c) for c in s0[show[0]]["cells"]]
        crops = crop_cells(renderer.render(ep, ep.start), cells, ep.grid_size, crop_margin)
        fig, axes = plt.subplots(2, len(cells), figsize=(1.9 * len(cells), 3.9),
                                 gridspec_kw={"height_ratios": [1.15, 1]}, squeeze=False)
        for j, (cell, crop) in enumerate(zip(cells, crops)):
            a = axes[0][j]
            a.imshow(crop); a.set_xticks([]); a.set_yticks([]); a.grid(False)
            obj = next(o for o in ep.objects if o.cell == cell)
            a.set_title(("TARGET\n" if cell == ep.target.cell else "\n") + obj.name, fontsize=8)
            b = axes[1][j]
            vals = [s0[p]["logits"][j] for p in show]
            b.bar(range(len(show)), vals, color=SLOTS[:len(show)], width=0.75)
            b.axhline(0, color=INK2, lw=0.8)
            b.set_xticks(range(len(show)), [p.split()[1] for p in show], fontsize=7.5)
            if j == 0:
                b.set_ylabel("SigLIP logit")
        ymin = min(min(s0[p]["logits"]) for p in show); ymax = max(max(s0[p]["logits"]) for p in show)
        for b in axes[1]:
            b.set_ylim(min(0, ymin) - 0.5, max(0, ymax) + 0.5)
        fig.suptitle(f"Episode #{eid}, first observation: candidate crops and their logits vs "
                     f"“{ep.instruction}” (x-axis = tokens)", fontsize=9.5)
        _save(fig, figures_dir, "fig6_candidate_scores", paths)
    return paths
