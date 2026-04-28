# Span Labeling with Large Language Models

Code accompanying the master's thesis *Span Labeling with Decoder-Only Large Language Models*. The project compares four span labeling formats for
decoder-only LLMs (XML tagging, JSON-based extraction, JSON with occurrence
indices, and positional indices) across NER, grammatical error correction
(MultiGEC), translation quality estimation (WMT24), and a synthetic pattern
matching benchmark. It also includes a custom constrained decoding logits
processor that forces generated spans to be exact substrings of the input.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management
and Python 3.12.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Copy `.env.example` to `.env` and fill in the provider credentials you intend
to use:

```bash
cp .env.example .env
```

Relevant variables:

- `OPENAI_API_KEY` for OpenAI models (e.g. `gpt-5-mini`).
- `OPENROUTER_API_KEY` if routing through OpenRouter.
- `VLLM_API_KEY`, `VLLM_PORT` for the local vLLM server. The default port is
  `8057`; SLURM jobs override it per-job (see below).
- `OLLAMA_PORT` if you want to test against a local Ollama server.

For local vLLM runs, model weights are read from `HF_HOME`. Pre-download the
models you plan to use.


## Running an experiment

Experiments are configured by YAML files under `span_labeling/configs/`. Each
config specifies the model, dataset groups, methods, seeds, and (for vLLM)
server flags. The runner expands the cartesian product of
`models x methods x datasets x seeds` and writes one JSON result file per
combination into `results/`.

There are two ways to run: submit to SLURM (intended path) or run locally
against an already-running provider.


### Submitting to SLURM

`span_labeling/jobs/submit.py` generates and submits a SLURM script from a
config. For vLLM configs it spins up the server on a job-local port, waits
for the health check, runs the experiment, and tears the server down.

```bash
uv run python -m span_labeling.jobs.submit \
    span_labeling/configs/qwen/qwen3_8b_ner_multigec.yaml
```

To inspect the generated script without submitting:

```bash
uv run python -m span_labeling.jobs.submit \
    span_labeling/configs/qwen/qwen3_8b_ner_multigec.yaml --dry_run
```

The `slurm:` block in each config controls partition, GPU count, GPU
constraint, time, and memory. Logs land in `logs/<job_name>_<job_id>.{out,err}`.


### Running locally

If a vLLM server is already running on `$PORT`, or the config uses a remote
provider (OpenAI), invoke the runner directly:

```bash
# Remote provider (no server needed)
uv run python -m span_labeling.experiments.run \
    span_labeling/configs/gpt_5_mini/gpt_5_mini_ner_multigec.yaml

# Local vLLM on port 8057
uv run python -m span_labeling.experiments.run \
    span_labeling/configs/qwen/qwen3_8b_ner_multigec.yaml --port 8057
```

The runner respects `project.skip_experiment_if_exists` (default true): if a
matching row already exists in `results/results.csv` for the same
experiment / model / method / dataset / seed / flags, that combination is
skipped. Set it to `false` in the config to force re-runs.


## Results

Each experiment writes `<experiment>_<model>_<method>_<dataset>_<seed>_<timestamp>_results.json`
into the output directory. The summary table with results from all experiment runs
reported in the thesis is here: `results/results.csv` —
the analysis notebooks and figures read directly from that file.

To regenerate the CSV from scratch (for example after adding new runs), use:

```bash
uv run python -m span_labeling.results
```

This reads every JSON in `results/` and writes `results/_results.csv` with
soft / hard F1, error breakdowns, and token usage. Constrained runs that
aren't in the `*_fixed` namespace are skipped automatically (the processor
in those earlier runs had a bug; the `constrained_fixed` configs are the
canonical version).


## Repository layout

```
span_labeling/
    base.py                  Method and dataset registries.
    config.py                Pydantic settings loaded from YAML + .env.
    dataset.py               Per-task dataset loaders.
    metrics.py               Character-level overlap F1 (soft and hard).
    error_analysis.py        Per-prediction error type classification.
    results.py               Aggregates result JSONs into results.csv.
    prompt_utils.py          Builds prompts from prompts/*.yaml.
    constrained_processor.py SubstringCopyLogitsProcessor for vLLM.

    methods/                 Span labeling methods (xml, json, json_occurrence, index).
    modeling/                vLLM and OpenAI client wrappers.
    prompts/                 Per-method, per-task prompt templates.
    configs/                 Experiment YAMLs grouped by model.
    datasets/                Data conversion + synthetic generator.
    experiments/run.py       Experiment runner.
    jobs/submit.py           SLURM script generator and submitter.
```


## Adding a new experiment

1. Drop a YAML into `span_labeling/configs/<model>/`. The easiest path is to
   copy an existing config and edit `experiment.name`, `models`,
   `dataset_groups`, and `methods`. Dataset groups (`ner_all`, `wmt_all`,
   `multigec_all`, `synthetic_all`, etc.) are defined in
   `span_labeling/datasets/dataset_groups.py`.
2. For local models, set the vLLM provider block (`tensor_parallel_size`,
   `max_model_len`, optional `logits_processor`, `reasoning_parser`).
3. For constrained decoding, set `method.constrained: true` and
   `providers.vllm.logits_processor: span_labeling.constrained_processor:SubstringCopyLogitsProcessor`.
4. Submit with `span_labeling/jobs/submit.py` or run directly against an
   existing server.


## Notes

- The `constrained_processor.py` module restricts decoded spans to exact
  substrings of the input and (when `allowed_labels` is provided) restricts
  label values to that set. It works with vLLM's v1 engine via the
  `vllm_xargs` channel; see `modeling/vllm_model.py` for how it is wired.
- All experiments use one-shot prompting. The single in-context example per
  task lives in `span_labeling/prompts/*.yaml`.
- Evaluation pools character-overlap counts within a dataset and computes one
  micro-averaged F1 from those counts. Macro-averaging across datasets within
  a task happens downstream in the analysis scripts.