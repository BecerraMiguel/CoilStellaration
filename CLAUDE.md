# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`coilgen`: learning-based generation of stellarator coilsets for the **CoilStellaration** benchmark
(Proxima Fusion; paper arXiv:2611.12689, NeurIPS 2026 Evaluations & Datasets; released 2026-11-11).
Owner: Miguel Becerra (Spanish-speaking; goal: professional fusion+AI portfolio, learn along the way).
The codebase is nearly empty (scaffolding + upstream submodule); most of this file is context.
Status and plan live in `docs/ROADMAP.md`; the running journal in `docs/BITACORA.md` (both Spanish).

## What CoilStellaration is

Second step after ConStellaration (NeurIPS 2025: ~158k quasi-isodynamic (QI) plasma boundaries +
equilibria, no coils). Stellarator design = stage 1 (plasma boundary / MHD equilibrium, via
VMEC++/DESC) + stage 2 (coils reproducing the field: boundary must be a flux surface, B.n ~ 0).
Coils <- field is nonlinear and non-unique. Vision: generative models for each component, then
joint optimization.

Coils are **filamentary**: each a closed 3D Fourier curve (x,y,z)(t) up to order 7 plus a current
(46 DOF/coil). Only the unique coils of one half period are stored (field-period + stellarator
symmetry). Constraints (normalized by minor radius a): min coil-plasma distance d_CP/a, min
coil-coil distance d_CC/a, max curvature a*kappa_max; physics metric: normalized field error
eps = |B.n|/|B| on the boundary (<~1% good).

### Dataset (HF `proxima-fusion/coilstellaration`, MIT)
- Pipeline: filtered ConStellaration QI boundaries -> sampled requirements (N_C in [4,8],
  d_CP/a in [0.5,1.0], d_CC/a in [0.3,0.7] (HF card says 0.1-0.7), a*kappa in [1.5,5.0]) -> REGCOIL
  warm start (~0.1 CPUh) -> DESC augmented-Lagrangian optimization (~1.3 CPUh) -> normalize R=1 m,
  currents rescaled to B0=1 T -> metrics.
- Scale: 14,496 boundaries; 266,549 samples; 177,341 DESC coilsets (66.5%); 138,765 after quality
  filter. Multiple coilsets per boundary under different requirements (~12 DESC / boundary).
- Split disjoint by boundary hash 95/5: 253,250 train / 13,299 eval. N_FP in {2,3,4,5}, NFP=3
  oversampled; retention drops with N_FP; combos (NFP,NC) = (3,7),(3,8),(5,8) empty.
- Subsets: `results` (default, 266,549 rows, 0.3 GB, start here), `coilsets` (2.3 GB),
  `requirements` (0.08 GB), `metrics` (213 GB, NEVER download wholesale). Boundaries join from
  `proxima-fusion/constellaration` via `constellaration_boundary_id`.

## The challenge (most important section)

**Task**: given boundary b + requirements r = (d_CC_min, d_CP_min, kappa_max, eps_max) + N_C
(+ N_FP), output coilset c = f(b, r) in ONE forward pass (amortize the optimization). Targets are the
metrics the reference DESC coilset achieved, so every instance is feasible by construction.

**Tracks**: `fixed_shape` (NFP=3, 5 coils/half-period; 53,206 train / 2,771 eval) and
`variable_shape` (NFP 2-5, 4-8 coils; 155,850 train / 8,058 eval). Scored independently.

**Strata**: tertiles of reference mean field error (`desc_metrics/normalized_field_error/mean`):
`tight` (<0.31%, hardest), `medium`, `loose` (>1.25%, easiest). Headline view is `All` (pooled).

**Scoring** (`benchmark/scoring.py`, `ScoringSettings`):
- violation v_i = max(0, (g_i - t_i)/max(|t_i|,1e-6)) for upper-bound metrics (curvature, field
  error); (t_i - g_i)/... for lower-bound (distances). g = achieved, t = target.
- soft score s_i = exp(-v_i/alpha_i); alpha = 0.10 (d_CC, d_CP, curvature), 0.25 (field error).
- instance score S = geometric mean of the 4 s_i (one failure kills it). Wrong coil count -> S=0;
  non-finite metric -> v=inf.
- aggregates: `mean` (paper headline), `median`, `p10`, `feasibility_rate_0p9` (S>=0.9), and
  `strict_feasibility` (all v_i <= 1e-3).
- **Leaderboard ranks by strict_feasibility THEN mean** (the paper ranks by mean). Achieved metrics
  are recomputed by DESC on the instance's own DESC equilibrium (`EvalData.desc_equilibrium`).

**ONNX submission contract** (public HF model repo, `model.onnx` <= 50 MiB):
- inputs: `boundary_r_cos`, `boundary_z_sin` float32 [n_pol,n_tor] (+`boundary_r_sin`,
  `boundary_z_cos` if non-symmetric); `requirement_metrics` float32 [4] in `RequirementMetrics`
  field order; `n_field_periods`, `n_coils_per_half_period` int64 scalars. Unscaled.
- outputs: `coil_x_n`, `coil_y_n`, `coil_z_n` [n_coils, 2N+1] in DESC `FourierXYZCoil` layout
  [sin N..sin 1, const, cos 1..cos N]; `currents` [n_coils]. n_coils must equal
  `n_coils_per_half_period`. Use dynamic axes.
- metadata_props `supported_tracks` = `fixed_shape` or `fixed_shape,variable_shape`.
- Leaderboard Space: `proxima-fusion/coilstellaration-bench` (+ evaluator Space, requests/results
  datasets). Source in `external/coilstellaration/leaderboard/`. ONNX export tooling:
  `leaderboard/scripts/export_coilstellaration_onnx.py` (needs onnx<1.23 + jax2onnx).

**Baselines** (paper): MLP family with bidirectional exponential spectral scaling
(coeff / 1.2^-q), d_h=512, L=4, AdamW, ~0.5 h on 32 CPUs, trained with supervised MSE on reference
coil coefficients (NO physics loss). Mean S / strict feasibility:
- fixed: EnsembleResMLP 0.133/4.4%; EnsembleMLP 0.111/2.7%; ResMLP 0.076/3.8%; MLP 0.067/2.6%.
- variable: EnsemblePerCoilMLP 0.137/3.1%; EnsembleMaskedMLP 0.128/2.9%; PerCoil 0.082/2.5%;
  Masked 0.081/2.2%.
- Tight stratum: 0% strict for all; medium <0.3%.

**Dominant failure mode: field error** (median violation 2.1-2.9, s_eps ~1e-5..1e-4). Ensembles
nail coil-plasma distance and curvature; coil-coil spacing often violated. Supervised coefficient
regression does not control B.n; physics-aware loss / refinement is the obvious gap.

**Warm-start result** (1,289 tight instances, 308 boundaries): learned start beats REGCOIL start for
up to 300 DESC iterations (100 learned its > 200 REGCOIL its; strict 22% at 200 vs REGCOIL 18% at
500); at 500 its REGCOIL mean 0.83 vs 0.76 but strict 18 vs 27%; at 1000 no gap (26 vs 25%).
Complementary weaknesses: REGCOIL coils well separated but too close to plasma; learned coils clear
plasma but too close to each other.

**Limitations** of the benchmark: filamentary coils (no finite build, forces, torsion, HTS strain,
ports, length bound); vacuum field on a fixed target boundary (not free-boundary); QI only; strata
defined only by field error.

### Live leaderboard findings (2026-11-11/12, our analysis)
- Proxima's best baseline on the live board: fixed mean ~0.177, strict ~5.1% (per_coil_ensemble).
- `phanerozoic/coilstellaration-circle-baseline`: planar circles of radius 0.3*R0, NO learning:
  fixed mean 0.117, strict 6.6% (beats every Proxima baseline on strict!); variable mean 0.089,
  strict 5.2%. The loose stratum contains trivially satisfiable instances and strict-first ranking
  rewards it. Always report any model against this circle baseline.
- `phanerozoic/coilstellaration-coil-optimizer`: single ONNX graph = (1) retrieval of nearest
  training coilsets (53k fixed_shape coilsets, fp16), (2) Proxima per-coil ensemble as seed,
  (3) Levenberg-Marquardt written with ONNX Loop ops, analytic Jacobians, ~150 steps, ~1.5-2.7 s per
  instance on one CPU. fixed mean 0.961, strict 78.8% (tight 65%).
- **Rules ambiguity**: "single forward pass" is only enforced via the ONNX contract, so an
  in-graph optimizer is allowed today; rules are likely to be clarified. A hybrid learned +
  in-graph refinement is clearly powerful; keep pure-learned and hybrid results separate and
  disclose which is which.

## Repository layout

- `src/coilgen/` our package (currently only `__version__`). Own code goes here, not in upstream.
- `external/coilstellaration/` upstream as a pinned git submodule (see `.gitmodules`), installed as
  an editable path dependency via `[tool.uv.sources]`. Never edit it in place; bump the pin
  deliberately and log it. Clone with `--recurse-submodules`.
- `tests/`, `scripts/`, `configs/`, `notebooks/`, `docs/` (`ROADMAP.md`, `BITACORA.md`, `papers/`).
- `data/` local-only (only `data/README.md` is tracked). `outputs/` and HF cache are gitignored.
  Upstream examples write to `~/tmp/outputs/coilstellaration/` (`paths.OUTPUTS_PATH`).

## Commands

```bash
uv sync                                   # Python 3.12; first sync is heavy (JAX, DESC, vmecpp)
uv run pytest tests/                      # our tests
uv run pytest tests/test_smoke.py::test_imports        # single test
uv run pytest external/coilstellaration/src/coilstellaration/benchmark/scoring_test.py   # cheap upstream test
uv run ruff check . && uv run ruff format .            # line length 100
uv run pre-commit run --all-files
JAX_PLATFORMS=cpu WANDB_MODE=offline uv run python external/coilstellaration/examples/run_model_training.py
uv run python external/coilstellaration/examples/run_benchmark_scoring.py --checkpoint-id DHmCARHZ4ay9rsMDPdjVCu3 --model-kind mlp --track fixed_shape --stratum tight --n 4
```
`regcoil` (Fortran binary) is only needed to GENERATE data (`run_regcoil.py`); it is not built here.
Upstream's own `pytest` config (`testpaths=["src","leaderboard/tests"]`) does not apply from this
root; pass paths explicitly.

## Upstream key modules (`external/coilstellaration/src/coilstellaration/`)

- `data_utils.py`: `load_benchmark_dataset(track, stratum, split, n)` (n=0 = all) returns `EvalData`
  rows; `load_ml_baseline_dataframe(track, split)` for training rows; `metrics_to_requirement_metrics`.
- `types.py`: `Coilset` (unique pre-symmetry coils, DESC `FourierXYZCoil`), `RequirementMetrics`,
  `EvalData`. `coilset_utils.coilstellaration_to_desc` converts to a DESC coilset.
- `metrics_utils_v2.evaluate_coilset_metrics(eq, coilset)`: the costly DESC evaluation used by scoring;
  `desc_utils.desc_equilibrium_to_desc_object` rebuilds the equilibrium.
- `benchmark/{scoring,scoring_types}.py`: exact scoring (`score_eval_data`, `score_benchmark`).
- `machine_learning/{model_definition,train}.py`: JAX/Flax NNX models; `TrainConfig.model_kind` in
  mlp, res_mlp, mlp_ensemble, res_mlp_ensemble (fixed) and per_coil, masked, per_coil_ensemble,
  masked_ensemble (variable); `predict_coilsets`, checkpoints via `read_checkpoint_from_id`.
  Bundled checkpoint: `mlp_DHmCARHZ4ay9rsMDPdjVCu3` (fixed_shape).
- `data_generation_tasks.optimize_coilset_using_desc`, `regcoil_utils`: data generation pipeline.
- `examples/`: run_regcoil, run_desc, run_model_training, run_model_prediction,
  run_benchmark_scoring (`--desc-maxiters` for warm-start experiments; official numbers only from
  the Space).

## Compute resources

- **Local** (2 CPU cores, 3 GB RAM total, often <1 GB free; no GPU, no Docker, no `gh`, no sudo/apt,
  ~145 GB disk): development, tests, light EDA on the `results` subset (load lazily with
  pyarrow/polars if `datasets` is heavy), tiny smoke runs. Not for training or full-eval DESC scoring.
- **Known local blocker**: the local CPU (Intel Celeron N2840) has no AVX; jaxlib wheels require it
  (`RuntimeError: ... built using AVX`; bypassing the guard gives SIGILL). So JAX, DESC and
  `import coilstellaration` do NOT work locally; upstream tests/scoring/training must run on Colab or
  university GPUs. Local-safe: pyarrow/pandas/HF download, `import coilgen`. `tests/test_smoke.py`
  skips the upstream import when AVX is missing.
- **Google Colab GPUs**: interactive experiments and prototyping.
- **University research-group GPUs**: the MAIN resource for training, sweeps and large DESC
  evaluations. Keep training scripts config-driven (`configs/`) so they run unchanged off-machine.
- Never download the `metrics` subset; avoid loading `coilsets` fully into RAM locally.

## Conventions

- Code, comments, commit messages, README, CLAUDE.md in English; `docs/ROADMAP.md` and
  `docs/BITACORA.md` in Spanish.
- Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `test:` ...). Every commit gets an entry
  in `docs/BITACORA.md`. Add explicit paths to `git add` (no `-A` / `.`). Miguel is always the
  commit author; Claude goes only as co-author via the footer
  `Co-Authored-By: <model> <noreply@anthropic.com>` (D-012).
- At session start read the latest `docs/BITACORA.md` entries; log each commit and decision (`D-NNN`) there.
- Data, checkpoints and ONNX files are never committed (`.gitignore`); paper PDFs stay local in
  `docs/papers/`.
- Ruff line length 100, Python 3.12, uv only (no pip installs).
- Model-selection policy for Claude subagents (`model` param: haiku|sonnet|opus|fable): see
  `docs/ROADMAP.md` for per-phase assignments. Defaults: Fable for deep research/novel reasoning,
  Opus for hard coding/design, Sonnet for routine coding/docs/EDA, Haiku for bulk/simple tasks.
