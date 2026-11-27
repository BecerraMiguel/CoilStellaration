"""In-process quick-check scoring of an upstream checkpoint on a few eval rows.

Same pipeline as the "End-to-end walkthrough" of upstream's README (load eval rows -> predict
coilsets -> DESC metrics on each row's own equilibrium -> benchmark score), run in a single
process so the JIT-compiled DESC evaluator is reused across instances.

Why not `external/coilstellaration/examples/run_benchmark_scoring.py`: at the pinned upstream
commit (5b925cf, and still on upstream main as of 2026-11-26) its per-instance worker imports
`coilstellaration.examples`, which is not part of the installed package, so every instance
fails and scores 0. See docs/BITACORA.md, Entrada 5.

Usage:
    uv run python scripts/quickcheck_score.py                       # bundled mlp, tight, n=4
    uv run python scripts/quickcheck_score.py --stratum loose --n 8

Local quick check only; official numbers come from the leaderboard Space.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

SHORT = {
    "min_normalized_coil_to_coil_distance": "d_CC",
    "min_normalized_coil_to_plasma_distance": "d_CP",
    "max_normalized_coil_curvature": "kappa",
    "max_normalized_field_error": "eps",
}


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint-id", default="DHmCARHZ4ay9rsMDPdjVCu3")
    parser.add_argument("--model-kind", default="mlp")
    parser.add_argument("--track", default="fixed_shape", choices=["fixed_shape", "variable_shape"])
    parser.add_argument("--stratum", default="tight", choices=["loose", "medium", "tight"])
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()

    import gc

    import jax
    from coilstellaration import coilset_utils, data_utils, desc_utils, metrics_utils_v2
    from coilstellaration.benchmark import scoring, scoring_types
    from coilstellaration.machine_learning import model_definition
    from coilstellaration.machine_learning import utils as ml_utils

    t_start = time.perf_counter()
    rows = data_utils.load_benchmark_dataset(
        track=args.track, stratum=args.stratum, split="eval", n=args.n
    )
    checkpoint = ml_utils.read_checkpoint_from_id(args.model_kind, args.checkpoint_id)
    model = model_definition.read_model_from_checkpoint(checkpoint)

    t0 = time.perf_counter()
    predictions = model_definition.predict_coilsets(model, rows)
    t_predict = time.perf_counter() - t0
    print(f"loaded {len(rows)} rows + model, predicted in {t_predict:.2f}s")

    # DESC evaluation peaks at ~3.7 GB; free the model and its compiled functions first so the
    # whole run fits on a 5.7 GB machine.
    del model, checkpoint, rows
    gc.collect()
    jax.clear_caches()

    settings = scoring_types.ScoringSettings()
    instance_scores = []
    print(f"\n{'#':>2} {'boundary':<25}{'S':>8}  violations v_i (d_CC d_CP kappa eps)   time")
    for i, row in enumerate(predictions):
        assert row.predicted_coilset is not None and row.desc_equilibrium is not None
        t0 = time.perf_counter()
        with warnings.catch_warnings(action="ignore"):
            metrics = metrics_utils_v2.evaluate_coilset_metrics(
                eq=desc_utils.desc_equilibrium_to_desc_object(row.desc_equilibrium),
                coilset=coilset_utils.coilstellaration_to_desc(row.predicted_coilset),
            )
        achieved = data_utils.metrics_to_requirement_metrics(metrics)
        score = scoring.score_eval_data(row, achieved, settings)
        instance_scores.append(score)
        v = score.metric_scores.violations
        vs = " ".join(f"{v[k]:.3g}" for k in SHORT)
        print(
            f"{i:>2} {row.boundary_id:<25}{score.aggregate:>8.2e}  {vs:<34}"
            f"{time.perf_counter() - t0:5.1f}s"
        )

    benchmark = scoring.score_benchmark(instance_scores, settings)
    print("\nSummary (local quick check, not official):")
    for name, value in benchmark.summaries.items():
        print(f"  {name:<22}{value:.4f}")
    print(f"\ntotal {time.perf_counter() - t_start:.0f}s | peak RSS {peak_rss_mb():.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
