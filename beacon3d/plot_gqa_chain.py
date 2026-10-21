#!/usr/bin/env python3
"""Grounding -> QA chain figures (Sankey) + JSON from Beacon3D's processed results.

For every setting in ``<result_dir>/<setting>/`` this reads
``infer_grounding_processed.json`` and ``infer_qa_processed.json`` and recomputes the four
shares exactly as ``eval_chain`` in Beacon3D's ``evaluate_qa.py``: an object's grounding
counts as correct only if all three of its chains end correct; a QA case counts as correct
if its GPT score is >= 4. "overall" uses every QA case, "wo_extra" only the cases without
``extra_knowledge``. Writes ``<result_dir>/figures/gqa_chain{,_wo_extra,_controls}.json``
and the PNG/PDF figures next to them.

    python beacon3d/plot_gqa_chain.py [--result_dir beacon3d/results]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DOMAINS = {"scannet": "ScanNet", "3rscan": "3RScan", "multiscan": "MultiScan"}
CONTROLS = {"scannet": "ScanNet", "scannet_myfeat": "ScanNet (own Uni3D + 2D)",
            "scannet_no2d": "ScanNet (no 2D token)"}
WITH_2D = {"scannet": "ScanNet", "3rscan_2d": "3RScan (+ 2D)", "multiscan_2d": "MultiScan (+ 2D)"}
BLUE, BLUE_FLOW, BLUE_CROSS = "#2878d6", "#8ab4e8", "#bfd5f1"
ORANGE, ORANGE_FLOW, ORANGE_CROSS = "#eb6a36", "#f2ad90", "#f7d2c2"
INK = "#1a1a1a"


def chain_shares(setting_dir):
    grounding = json.load(open(os.path.join(setting_dir, "infer_grounding_processed.json")))
    qa = json.load(open(os.path.join(setting_dir, "infer_qa_processed.json")))
    obj_ok = [all(grounding[i + k]["correct"][-1] for k in range(3))
              for i in range(0, len(grounding), 3)]
    out = {}
    for name, keep in [("overall", lambda it: True), ("wo_extra", lambda it: not it["extra_knowledge"])]:
        c = np.zeros(4)  # g1q1, g1q0, g0q1, g0q0
        for i, it in enumerate(qa):
            if keep(it):
                c[(0 if obj_ok[i // 3] else 2) + (0 if it["score"] >= 4 else 1)] += 1
        n = int(c.sum())
        out[name] = {"n": n, **{k: round(float(v) / n * 100, 2)
                                for k, v in zip(["g1q1", "g1q0", "g0q1", "g0q0"], c)}}
    return out


def band(ax, x0, x1, y0a, y0b, y1a, y1b, color):
    """Flow from [y0a, y0b] at x0 to [y1a, y1b] at x1 (smoothstep edges)."""
    t = np.linspace(0, 1, 100)
    s = t * t * (3 - 2 * t)
    x = x0 + (x1 - x0) * t
    ax.fill_between(x, y0a + (y1a - y0a) * s, y0b + (y1b - y0b) * s, color=color, lw=0)


def sankey(ax, d, title):
    g1q1, g1q0, g0q1, g0q0 = (d[k] / 100 for k in ["g1q1", "g1q0", "g0q1", "g0q0"])
    gap, w, x0, x1 = 0.03, 0.045, 0.0, 1.0
    scale = 1 - gap
    gc, qc = (g1q1 + g1q0) * scale, (g1q1 + g0q1) * scale
    # top of the axis is y = 1; nodes stack downwards
    lg = [(1, 1 - gc), (1 - gc - gap, 0)]          # grounding correct / incorrect
    rq = [(1, 1 - qc), (1 - qc - gap, 0)]          # QA correct / incorrect
    # straight flows first, crossings on top
    flows = [
        (lg[0][0], g1q1, rq[0][0], BLUE_FLOW),
        (lg[1][0] - g0q1 * scale, g0q0, rq[1][0] - g1q0 * scale, ORANGE_FLOW),
        (lg[1][0], g0q1, rq[0][0] - g1q1 * scale, ORANGE_CROSS),
        (lg[0][0] - g1q1 * scale, g1q0, rq[1][0], BLUE_CROSS),
    ]
    for ytop_l, share, ytop_r, color in flows:
        h = share * scale
        band(ax, x0 + w, x1 - w, ytop_l, ytop_l - h, ytop_r, ytop_r - h, color)
    for (top, bot), col in zip(lg, [BLUE, ORANGE]):
        ax.add_patch(plt.Rectangle((x0, bot), w, top - bot, color=col, lw=0))
    for (top, bot), col in zip(rq, [BLUE, ORANGE]):
        ax.add_patch(plt.Rectangle((x1 - w, bot), w, top - bot, color=col, lw=0))
    kw = dict(color=INK, fontsize=15, fontweight="bold", va="center", linespacing=1.15)
    ax.text(x0 + w + 0.02, (lg[0][0] + lg[0][1]) / 2, f"Grounding: Correct\n{(g1q1 + g1q0) * 100:.0f}%", **kw)
    ax.text(x0 + w + 0.02, (lg[1][0] + lg[1][1]) / 2, f"Grounding: Incorrect\n{(g0q1 + g0q0) * 100:.0f}%", **kw)
    ax.text(x1 - w - 0.02, (rq[0][0] + rq[0][1]) / 2, f"QA: Correct\n{(g1q1 + g0q1) * 100:.0f}%", ha="right", **kw)
    ax.text(x1 - w - 0.02, (rq[1][0] + rq[1][1]) / 2, f"QA: Incorrect\n{(g1q0 + g0q0) * 100:.0f}%", ha="right", **kw)
    # crossing shares, labelled near their own end of the crossing
    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    yb = lg[0][0] - g1q1 * scale - g1q0 * scale / 2
    ax.text(0.3, yb - 0.05 * (yb - (rq[1][0] - g1q0 * scale / 2)), f"{g1q0 * 100:.0f}%",
            fontsize=14, color=INK, ha="center", va="center", bbox=box)
    yo = rq[0][0] - g1q1 * scale - g0q1 * scale / 2
    ax.text(0.72, yo + 0.25 * ((lg[1][0] - g0q1 * scale / 2) - yo), f"{g0q1 * 100:.0f}%",
            fontsize=14, color=INK, ha="center", va="center", bbox=box)
    ax.set_title(f"{title}  (n = {d['n']} QA cases)", loc="left", fontsize=16, color=INK, pad=14)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.axis("off")


def save(fig, path):
    fig.savefig(path + ".png", dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(path + ".pdf", bbox_inches="tight", facecolor="white", metadata={"CreationDate": None})
    plt.close(fig)


def panels(data, names, path):
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 4))
    for ax, (key, title) in zip(np.atleast_1d(axes), names.items()):
        sankey(ax, data[key], title)
    fig.tight_layout()
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--result_dir", default="beacon3d/results")
    args = ap.parse_args()
    fig_dir = os.path.join(args.result_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    settings = {**DOMAINS, **CONTROLS}
    has_2d = all(os.path.exists(os.path.join(args.result_dir, s, "infer_qa_processed.json")) for s in WITH_2D)
    if has_2d:
        settings.update(WITH_2D)
    shares = {s: chain_shares(os.path.join(args.result_dir, s)) for s in settings}

    for name, keys, part in [("gqa_chain", DOMAINS, "overall"), ("gqa_chain_wo_extra", DOMAINS, "wo_extra"),
                             ("gqa_chain_controls", CONTROLS, "overall")]:
        json.dump({k: shares[k][part] for k in keys}, open(os.path.join(fig_dir, f"{name}.json"), "w"), indent=1)
    for key, title in DOMAINS.items():
        panels({key: shares[key]["overall"]}, {key: title}, os.path.join(fig_dir, f"gqa_chain_{key}"))
        panels({key: shares[key]["wo_extra"]}, {key: title}, os.path.join(fig_dir, f"gqa_chain_{key}_wo_extra"))
    for key, title in CONTROLS.items():
        panels({key: shares[key]["overall"]}, {key: title}, os.path.join(fig_dir, f"gqa_chain_{key}_controls"))
    panels({k: shares[k]["overall"] for k in DOMAINS}, DOMAINS, os.path.join(fig_dir, "gqa_chain_all"))
    panels({k: shares[k]["wo_extra"] for k in DOMAINS}, DOMAINS, os.path.join(fig_dir, "gqa_chain_all_wo_extra"))
    panels({k: shares[k]["overall"] for k in CONTROLS}, CONTROLS, os.path.join(fig_dir, "gqa_chain_all_controls"))
    if has_2d:  # 3RScan / MultiScan with our DINOv2 2D token (extract_videofeats_gt.py)
        json.dump({k: shares[k]["overall"] for k in WITH_2D}, open(os.path.join(fig_dir, "gqa_chain_2d.json"), "w"), indent=1)
        json.dump({k: shares[k]["wo_extra"] for k in WITH_2D}, open(os.path.join(fig_dir, "gqa_chain_2d_wo_extra.json"), "w"), indent=1)
        for key in ["3rscan_2d", "multiscan_2d"]:
            panels({key: shares[key]["overall"]}, {key: WITH_2D[key]}, os.path.join(fig_dir, f"gqa_chain_{key}"))
        panels({k: shares[k]["overall"] for k in WITH_2D}, WITH_2D, os.path.join(fig_dir, "gqa_chain_all_2d"))
        panels({k: shares[k]["wo_extra"] for k in WITH_2D}, WITH_2D, os.path.join(fig_dir, "gqa_chain_all_2d_wo_extra"))
    print(json.dumps({k: v["overall"] for k, v in shares.items()}, indent=1))


if __name__ == "__main__":
    main()
