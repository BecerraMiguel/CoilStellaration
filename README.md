# coilgen

Learning-based generation of stellarator coilsets for the
[CoilStellaration](https://arxiv.org/abs/2611.12689) benchmark by Proxima Fusion.

Given a plasma boundary and engineering requirements (coil-coil distance, coil-plasma distance,
maximum curvature, field-error budget), predict a filamentary coilset in a single forward pass,
exported as an ONNX model for the public leaderboard.

## Status

Early stage: repository scaffolding, environment and data access. See `docs/ROADMAP.md`
(Spanish) for the plan.

## Structure

```
src/coilgen/            our package (models, training, ONNX export)
external/coilstellaration  upstream repo (git submodule, pinned, editable dependency)
tests/  scripts/  configs/  notebooks/
docs/                   roadmap, project journal, paper notes (PDFs local only)
data/                   local-only, gitignored
```

## Setup

```bash
git clone --recurse-submodules <this-repo>
cd CoilStellaration
uv sync                      # Python 3.12, installs upstream editable + dev tools
uv run pytest tests/
uv run pre-commit install    # optional
```

The REGCOIL Fortran binary is only needed to generate data and is not built by default
(see the upstream repo for build instructions).

## Links

- Paper: https://arxiv.org/abs/2611.12689
- Dataset: https://huggingface.co/datasets/proxima-fusion/coilstellaration
- Upstream code: https://github.com/proximafusion/coilstellaration
- Leaderboard: https://huggingface.co/spaces/proxima-fusion/coilstellaration-bench

## Acknowledgements

The dataset, benchmark and upstream code are by Proxima Fusion (MIT license). This project
builds on them; all credit for the benchmark goes to the original authors.

## License

MIT, see `LICENSE`.
