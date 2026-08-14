"""The binary-stratification null, as a general result rather than a case study.

This formalises the trap that produced this project's one retraction. It is a
STRESS TEST, not a measurement of any model: every number in here is generated
by a simulated responder that has **no item-level information whatsoever**.

**The identity.** For binary ground truth `y` and a model prediction `M`,
correctness is `C = 1[M = y]`. Within a stratum where `y` is constant:

    y = 1  =>  C = M
    y = 0  =>  C = 1 - M

so correctness is a *relabelling of the prediction*, carrying no independent
information. That alone is arithmetic. The trap needs one further step, and it
is the step that makes the effect appear:

**the uncertainty score is computed from the same K votes that produce `M` by
majority.** Vote entropy is therefore a deterministic function of the vote
count `S = sum_j V_ij`, and so is `M`, and so -- inside a fixed-label stratum
-- is `C`. A score built from the votes and a label built from the votes are
not independent, and ranking one by the other measures the voting arithmetic.

Concretely with `V_ij ~ Bernoulli(p)` i.i.d. and odd `K`: `S ~ Binomial(K, p)`
is the *only* random quantity per item. When `p` is far from `0.5` the majority
is usually right within the favoured stratum, and the rare items where it is
wrong are exactly the ones whose votes were split -- that is, the high-entropy
ones. Entropy then "predicts" error with AUROC well above chance while
containing no information about any item.

Because everything is a function of `S`, the AUROC has a closed form
(`exact_stratum_auroc`) as well as a Monte Carlo estimate (`simulate_stratum`).
The exact value is the oracle the simulation is tested against; the simulation
is what gets displayed, because it is the thing a practitioner would actually
run and it carries the sampling spread.

Nothing here calls a model, touches a scorer rule, or reads a result CSV.
`plotting.bias_only_null_auroc` is deliberately left untouched.
"""

import hashlib
import json
import math
import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import plotting

#: Odd K only. An even K makes the majority ill-defined on a tie, and the
#: tie-breaking convention would then silently drive the result.
DEFAULT_K = (3, 5, 7, 9, 15)

#: 0.05 .. 0.95. p is the per-sample probability of the responder emitting a
#: "1" vote; it is identical for every item, which is what makes it a null.
DEFAULT_P = tuple(round(0.05 * i, 2) for i in range(1, 20))

TRUTH_CONDITIONS = ("y=1", "y=0", "pooled_balanced")


def binary_entropy(frac: np.ndarray) -> np.ndarray:
    """Shannon entropy in NATS of a Bernoulli(frac), matching cluster entropy.

    Nats rather than bits so the scale is comparable to the project's K=5
    cluster entropy, whose ceiling is ln(5). `0 log 0` is taken as 0.
    """
    f = np.asarray(frac, dtype=float)
    out = np.zeros_like(f)
    mid = (f > 0) & (f < 1)
    fm = f[mid]
    out[mid] = -(fm * np.log(fm) + (1 - fm) * np.log(1 - fm))
    return out


def vote_entropy(counts: np.ndarray, k: int) -> np.ndarray:
    """Entropy of a vote count, folded on the INTEGER so `s` and `K-s` tie.

    Use this, not `binary_entropy(s/k)`, wherever ties matter -- which here is
    everywhere. `H(f) == H(1-f)` mathematically, but `3/7` and `1 - 4/7` are
    different floats, so evaluating the two ends separately gave entropies
    differing in the last bits and SILENTLY BROKE THE TIES. Ties are not an
    edge case in this simulation; they carry most of the mass. The symptom was
    the p=0.5 AUROC coming out at 0.494 for K=7 when symmetry requires exactly
    0.5, and a test caught it. Folding `m = min(s, K-s)` before dividing makes
    both ends land on the identical float by construction.
    """
    m = np.minimum(np.asarray(counts, dtype=np.int64), k - np.asarray(counts, dtype=np.int64))
    return binary_entropy(m / k)


def _auroc(entropy: np.ndarray, correct: np.ndarray) -> float:
    """AUROC for entropy predicting ERROR, via `plotting.compute_auroc`.

    Routed through the project's own function rather than reimplemented, so
    the orientation and the average-rank tie handling are identical to every
    other AUROC in this repository. Ties are the norm here, not an edge case:
    at K=5 the vote entropy takes only 4 distinct values.
    """
    df = pd.DataFrame({"e": np.asarray(entropy, float),
                       "c": np.asarray(correct, bool)})
    return plotting.compute_auroc(df, "e", "c")


# --- the closed form -------------------------------------------------------

def _vote_table(k: int, p: float) -> tuple:
    """(counts 0..K, their binomial probabilities, entropy, majority)."""
    s = np.arange(k + 1)
    logpmf = (np.array([math.lgamma(k + 1) - math.lgamma(i + 1)
                        - math.lgamma(k - i + 1) for i in s])
              + s * np.log(p) + (k - s) * np.log1p(-p))
    return s, np.exp(logpmf), vote_entropy(s, k), (s > k / 2).astype(int)


def exact_stratum_auroc(p: float, k: int, y: int) -> dict:
    """Closed-form AUROC inside a fixed-label stratum. The oracle.

    Everything per item is a function of the vote count `S ~ Binomial(K, p)`,
    so the joint law of (entropy, correctness) is a (K+1)-cell table and the
    Mann-Whitney AUROC is an exact double sum over it, ties at one half.
    """
    if k % 2 == 0:
        raise ValueError(f"K must be odd; got {k}")
    if y not in (0, 1):
        raise ValueError(f"y must be 0 or 1; got {y}")
    s, pmf, ent, maj = _vote_table(k, p)
    correct = (maj == y)
    p_cor, p_err = pmf[correct].sum(), pmf[~correct].sum()
    if p_cor <= 0 or p_err <= 0:
        return {"auroc": float("nan"), "p_correct_class": float(p_cor),
                "p_error_class": float(p_err), "defined": False}
    e_cor, w_cor = ent[correct], pmf[correct] / p_cor
    e_err, w_err = ent[~correct], pmf[~correct] / p_err
    greater = ((e_err[:, None] > e_cor[None, :]) * (w_err[:, None] * w_cor[None, :])).sum()
    equal = ((e_err[:, None] == e_cor[None, :]) * (w_err[:, None] * w_cor[None, :])).sum()
    return {"auroc": float(greater + 0.5 * equal),
            "p_correct_class": float(p_cor), "p_error_class": float(p_err),
            "defined": True}


# --- the simulation --------------------------------------------------------

def simulate_stratum(p: float, k: int, y, n_items: int = 650,
                     n_sims: int = 2000, seed: int = 0) -> dict:
    """Monte Carlo AUROC for a signal-free responder, one grid cell.

    `y` is 0, 1, or the string `pooled_balanced` for a half-and-half sample --
    the sanity condition, where the two strata's effects cancel and the AUROC
    must return to chance. A replicate with only one correctness class yields
    an UNDEFINED AUROC and is counted, never silently replaced by 0.5.
    """
    if k % 2 == 0:
        raise ValueError(f"K must be odd; got {k}")
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    if y == "pooled_balanced":
        y_vec = np.zeros(n_items, dtype=int)
        y_vec[: n_items // 2] = 1
    else:
        y_vec = np.full(n_items, int(y), dtype=int)

    aurocs, minority = [], []
    n_invalid = 0
    for _ in range(n_sims):
        # Sufficient statistic: only the vote COUNT matters, so draw it
        # directly instead of materialising n_items x K Bernoulli draws.
        s = rng.binomial(k, p, size=n_items)
        maj = (s > k / 2).astype(int)
        ent = vote_entropy(s, k)
        correct = (maj == y_vec)
        n_cor = int(correct.sum())
        minority.append(min(n_cor, n_items - n_cor))
        if n_cor == 0 or n_cor == n_items:
            n_invalid += 1
            continue
        aurocs.append(_auroc(ent, correct))
    a = np.asarray(aurocs, dtype=float)
    a = a[~np.isnan(a)]
    return {
        "truth_condition": "pooled_balanced" if y == "pooled_balanced" else f"y={y}",
        "p": float(p), "k": int(k), "n_items": int(n_items),
        "n_sims": int(n_sims), "seed": int(seed),
        "n_valid": int(a.size), "n_invalid": int(n_invalid),
        "auroc_median": float(np.median(a)) if a.size else float("nan"),
        "auroc_p2.5": float(np.percentile(a, 2.5)) if a.size else float("nan"),
        "auroc_p97.5": float(np.percentile(a, 97.5)) if a.size else float("nan"),
        "minority_median": float(np.median(minority)) if minority else float("nan"),
        "minority_rate_median": (float(np.median(minority)) / n_items
                                 if minority else float("nan")),
        "runtime_s": round(time.perf_counter() - t0, 3),
    }


def _cell_seed(base: int, condition: str, p: float, k: int) -> int:
    """A deterministic, independent seed per grid cell.

    Derived by hashing the cell's coordinates rather than incrementing a
    counter, so adding a p or K later does not renumber -- and therefore does
    not silently change -- every other cell's stream.
    """
    key = f"{base}|{condition}|{p:.4f}|{k}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def null_grid(p_values: Sequence[float] = DEFAULT_P,
              k_values: Sequence[int] = DEFAULT_K,
              conditions: Sequence[str] = TRUTH_CONDITIONS,
              n_items: int = 650, n_sims: int = 2000, base_seed: int = 20260814,
              smoke: bool = False, progress: bool = False) -> pd.DataFrame:
    """The full grid, with the exact oracle alongside every simulated cell."""
    rows = []
    cells = [(c, p, k) for c in conditions for p in p_values for k in k_values]
    it = cells
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(cells, desc="null grid")
        except ImportError:
            pass
    for cond, p, k in it:
        y = "pooled_balanced" if cond == "pooled_balanced" else int(cond.split("=")[1])
        seed = _cell_seed(base_seed, cond, p, k)
        row = simulate_stratum(p, k, y, n_items=n_items, n_sims=n_sims, seed=seed)
        row["smoke"] = bool(smoke)
        if cond == "pooled_balanced":
            row["auroc_exact"] = float("nan")
            row["exact_available"] = False
        else:
            ex = exact_stratum_auroc(p, k, y)
            row["auroc_exact"] = ex["auroc"]
            row["exact_available"] = bool(ex["defined"])
        rows.append(row)
    return pd.DataFrame(rows)


def collapse_identity(k: int = 5, p: float = 0.8, n_items: int = 64,
                      seed: int = 0) -> dict:
    """Demonstrate `C = M` when y=1 and `C = 1-M` when y=0, on actual draws.

    Asserting the identity in prose is cheap; showing it holds elementwise on
    simulated votes is what makes the proposition checkable.
    """
    rng = np.random.default_rng(seed)
    s = rng.binomial(k, p, size=n_items)
    maj = (s > k / 2).astype(int)
    return {
        "k": k, "p": p, "n_items": n_items,
        "y1_correct_equals_prediction": bool(np.array_equal((maj == 1), maj.astype(bool))),
        "y0_correct_equals_complement": bool(np.array_equal((maj == 0), (1 - maj).astype(bool))),
        "n_distinct_entropy_values": int(np.unique(vote_entropy(s, k)).size),
    }


def write_grid(out_dir: str, grid: pd.DataFrame, config: dict) -> dict:
    """Write `null_grid.csv` + `null_grid_config.json` and hash both."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "null_grid.csv")
    cfg_path = os.path.join(out_dir, "null_grid_config.json")
    grid.to_csv(csv_path, index=False)
    with open(cfg_path, "w") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
    return {p: sha256_file(p) for p in (csv_path, cfg_path)}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: The three already-locked K=5 cases, for overlay. Values are the frozen
#: constants from `pilot/tests/test_stratum_degeneracy.py`; the observed AUROC
#: is what the MODEL scored, the curve is what a signal-free responder scores.
LOCKED_CASES = (("Qwen2.5-VL-3B", 650, 0.843, 0.854),
                ("Qwen2.5-VL-7B", 648, 0.813, 0.801),
                ("LLaVA-NeXT-7B", 400, 0.780, 0.775))


def plot_null_grid(grid: pd.DataFrame, path: str,
                   locked: Sequence = LOCKED_CASES) -> str:
    """Faceted null curves with a chance line and the three locked cases.

    The point the figure has to make is not that the curve is high, but that
    the MODEL'S OWN observed values sit ON OR BELOW it. So the locked cases are
    plotted as points against the y=1 facet rather than tabulated elsewhere.
    """
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conds = [c for c in TRUTH_CONDITIONS if c in set(grid["truth_condition"])]
    fig, axes = plt.subplots(1, len(conds), figsize=(4.1 * len(conds), 3.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    ks = sorted(grid["k"].unique())
    cmap = plt.get_cmap("viridis")
    for ax, cond in zip(axes, conds):
        sub = grid[grid["truth_condition"] == cond]
        for n, k in enumerate(ks):
            s = sub[sub["k"] == k].sort_values("p")
            col = cmap(n / max(1, len(ks) - 1))
            ax.plot(s["p"], s["auroc_median"], color=col, lw=1.6,
                    label=f"K={k}", zorder=3)
            ax.fill_between(s["p"], s["auroc_p2.5"], s["auroc_p97.5"],
                            color=col, alpha=0.12, lw=0, zorder=2)
        ax.axhline(0.5, color="0.35", ls="--", lw=1.0, zorder=1)
        ax.set_xlabel("per-sample vote rate $p$")
        ax.set_title({"y=1": "stratum $y=1$", "y=0": "stratum $y=0$",
                      "pooled_balanced": "balanced pooled"}.get(cond, cond),
                     fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25, lw=0.5)
        if cond == "y=1":
            # The three locked cases sit within 0.06 of each other in p, so
            # inline annotations overlapped into an unreadable smear -- caught
            # by rendering the figure and looking at it, not by an assert.
            # Points on the axes, names in a separate stacked block well clear
            # of the curves.
            for name, _n, pv, obs in locked:
                ax.plot([pv], [obs], marker="o", ms=6, color="#c0392b",
                        zorder=5, mec="white", mew=0.9)
            block = "\n".join(f"{n}: p={pv:.3f}, observed {o:.3f}"
                               for n, _x, pv, o in locked)
            ax.text(0.03, 0.97, "observed stratified AUROC\n" + block,
                    transform=ax.transAxes, fontsize=5.8, color="#c0392b",
                    va="top", ha="left", linespacing=1.45,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="#c0392b", lw=0.6, alpha=0.92), zorder=6)
    axes[0].set_ylabel("AUROC, entropy predicting error")
    axes[-1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("A responder with NO item-level information still scores far "
                 "above chance inside a fixed-label stratum", fontsize=9.5,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path
