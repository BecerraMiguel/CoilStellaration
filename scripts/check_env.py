"""Environment and pipeline sanity check for coilgen.

Loads a few rows of the CoilStellaration eval split, re-evaluates the *reference* DESC coilset
of each row with upstream's DESC metrics (the exact code path the leaderboard uses), and checks
that the achieved metrics reproduce the row's target `RequirementMetrics`.

The targets are, by construction, the metrics the reference coilset achieved, so a healthy
environment must give near-zero relative error and a perfect benchmark score (S = 1, strictly
feasible). It also reports wall time and peak memory per stage, which is the cost of one DESC
evaluation on this machine.

Usage:
    uv run python scripts/check_env.py                      # 1 row, fixed_shape, tight
    uv run python scripts/check_env.py --n 3 --stratum loose
    uv run python scripts/check_env.py --track variable_shape --n 2

Exit code is 0 if every row passes, 1 otherwise. Needs a CPU with AVX (JAX); downloads data
from the Hugging Face Hub on first run.
"""

from __future__ import annotations

import argparse
import os
import platform
import resource
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

METRIC_NAMES = (
    "min_normalized_coil_to_coil_distance",
    "min_normalized_coil_to_plasma_distance",
    "max_normalized_coil_curvature",
    "max_normalized_field_error",
)
SHORT = {
    "min_normalized_coil_to_coil_distance": "d_CC/a",
    "min_normalized_coil_to_plasma_distance": "d_CP/a",
    "max_normalized_coil_curvature": "a*kappa_max",
    "max_normalized_field_error": "eps_max",
}


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MB (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def cpu_has_avx() -> bool:
    try:
        return " avx " in f" {Path('/proc/cpuinfo').read_text()} ".replace("\n", " ")
    except OSError:
        return True


def print_environment() -> None:
    import desc
    import jax

    import coilgen

    print("== Environment")
    print(f"python        {platform.python_version()} ({sys.executable})")
    print(f"platform      {platform.platform()}")
    print(f"cpu has AVX   {cpu_has_avx()}")
    print(f"jax           {jax.__version__}  devices={jax.devices()}")
    print(f"desc          {desc.__version__}")
    print(f"coilgen       {coilgen.__version__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--track", default="fixed_shape", choices=["fixed_shape", "variable_shape"])
    parser.add_argument("--stratum", default="tight", choices=["loose", "medium", "tight"])
    parser.add_argument("--n", type=int, default=1, help="number of eval rows to check")
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="max relative error per metric (1e-3 = the benchmark's strict tolerance)",
    )
    args = parser.parse_args()

    if not cpu_has_avx():
        print("No AVX on this CPU: jaxlib cannot run. Use the Ryzen PC, Colab or the cluster.")
        return 1

    print_environment()

    from coilstellaration import coilset_utils, data_utils, desc_utils, metrics_utils_v2
    from coilstellaration.benchmark import scoring, scoring_types

    print(f"\n== Loading {args.n} eval row(s): track={args.track} stratum={args.stratum}")
    t0 = time.perf_counter()
    rows = data_utils.load_benchmark_dataset(
        track=args.track, stratum=args.stratum, split="eval", n=args.n
    )
    print(f"loaded in {time.perf_counter() - t0:.1f}s | peak RSS {peak_rss_mb():.0f} MB")

    settings = scoring_types.ScoringSettings()
    all_ok = True
    for i, row in enumerate(rows):
        assert row.desc_equilibrium is not None, "eval rows must carry their DESC equilibrium"
        target = row.requirement_metrics
        print(
            f"\n== Row {i}: boundary {row.boundary_id} | "
            f"N_FP={row.boundary.n_field_periods} N_C={target.n_coils_per_half_period}"
        )

        t0 = time.perf_counter()
        eq = desc_utils.desc_equilibrium_to_desc_object(row.desc_equilibrium)
        coilset = coilset_utils.coilstellaration_to_desc(row.true_coilset)
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        with warnings.catch_warnings(action="ignore"):
            metrics = metrics_utils_v2.evaluate_coilset_metrics(eq=eq, coilset=coilset)
        t_eval = time.perf_counter() - t0
        achieved = data_utils.metrics_to_requirement_metrics(metrics)

        print(f"{'metric':<12}{'target':>14}{'achieved':>14}{'rel. error':>12}")
        worst = 0.0
        for name in METRIC_NAMES:
            t, g = float(getattr(target, name)), float(getattr(achieved, name))
            rel = abs(g - t) / max(abs(t), 1e-12)
            worst = max(worst, rel)
            print(f"{SHORT[name]:<12}{t:>14.6g}{g:>14.6g}{rel:>12.2e}")

        score = scoring.score_eval_data(row, achieved, settings)
        ok = (
            worst <= args.rtol
            and int(achieved.n_coils_per_half_period) == int(target.n_coils_per_half_period)
            and score.metric_scores.strictly_feasible
        )
        all_ok &= ok
        print(
            f"score S={score.aggregate:.4f} strict={score.metric_scores.strictly_feasible} "
            f"| worst rel. error {worst:.2e} -> {'PASS' if ok else 'FAIL'}"
        )
        print(
            f"time: build eq+coils {t_build:.1f}s, DESC metrics {t_eval:.1f}s "
            f"| peak RSS {peak_rss_mb():.0f} MB"
        )

    print(f"\n== {'ALL PASS' if all_ok else 'SOME ROWS FAILED'} ({len(rows)} row(s))")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
