# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `sam3/`: `model_builder.py` wires image/video predictors, `model/` holds architectures, `agent/` wraps LLM-assisted prompting, `train/` contains Hydra configs plus training/eval logic, `eval/` covers metrics, and `perflib/` includes performance utils and tests.
- User-facing samples are `examples/` (notebooks) and `inference.py` (quick image/video demo). Media is under `assets/`; result-parsing helpers sit in `scripts/`.

## Build, Test, and Development Commands
- Install in editable mode: `pip install -e .`; add extras as needed: `pip install -e ".[dev]"`, `pip install -e ".[train]"`, `pip install -e ".[notebooks]"`.
- Authenticate before fetching checkpoints: `hf auth login` (Hugging Face gated models).
- Run a minimal demo once weights are available: `python inference.py` (set image/video paths and prompts).
- Launch training/eval with Hydra configs, e.g. `python sam3/train/train.py -c sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0`.
- Tests are light today; run current coverage with `pytest sam3/perflib/tests -q`. Add new suites under `tests/` (`test_*.py`) and keep them GPU-optional.

## Coding Style & Naming Conventions
- Python formatting follows Black (line length 88) with Black/`usort` import ordering; run `ufmt format sam3` from the `dev` extra. Prefer type hints; mypy config (strict on untyped defs) applies to new modules.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and descriptive Hydra config names in `sam3/train/configs/` (dataset_model_detail.yaml). Keep public entry points in `model_builder.py` stable.

## Testing Guidelines
- Mirror the pytest pattern (`Test*` classes, `test_*` functions`). If adding CUDA-dependent tests, guard them with `pytest.importorskip` or `pytest.mark.skipif` so CPU-only runs still pass.
- For data pipelines or configs, add lightweight sanity checks (shape/metadata assertions); prefer tiny synthetic fixtures over real data.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative summaries (e.g., `Fix SA-CO benchmark link`, optionally suffixed with PR number like `(#149)`); keep subject lines under ~72 chars.
- PRs should describe scope, configs used, and expected outputs; link issues, list validation commands (formatting, pytest), and include screenshots or log snippets for user-visible changes. Do not commit checkpoints, large artifacts, or private dataset paths; document required environment variables instead.

## Security & Configuration Tips
- Never hardcode tokens or dataset locations; rely on env vars or config entries under `sam3/train/configs/`. Keep sample commands pointed at local test data only.
- Large weight downloads are gated; reference the Hugging Face repo rather than mirroring binaries in-tree.
