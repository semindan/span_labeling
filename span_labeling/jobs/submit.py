import subprocess
from pathlib import Path
import fire
from span_labeling.config import CONFIGS_DIR, Settings, VllmSettings


def build_vllm_cmd(v: VllmSettings) -> str:
    parts = [
        f"uv run vllm serve {v.model}",
        "    --port $PORT",
        "    --seed 0",
        "    --attention-backend FLASH_ATTN",
        f"    --max-num-seqs {v.max_num_seqs}",
        f"    --gpu-memory-utilization {v.gpu_mem_util}",
        f"    --max-model-len {v.max_model_len}",
        f"    --max-num-batched-tokens {v.max_batched_tokens}",
        f"    --dtype {v.dtype}",
        f"    --tensor-parallel-size {v.tensor_parallel_size}",
        "    --trust-remote-code",
        # f"    --disable-log-requests",
        # f"    --enable-chunked-prefill",
    ]
    if v.logits_processor:
        parts.append(f"    --logits_processors {v.logits_processor}")
    if v.quantization:
        parts.append(f"    --quantization {v.quantization}")
    if v.chat_template:
        parts.append(f"    --chat-template {v.chat_template}")
    if v.reasoning_parser:
        parts.append(f"    --reasoning-parser {v.reasoning_parser}")
    return " \\\n".join(parts)


def generate_vllm_block(cfg: Settings) -> str:
    vllm_cmd = build_vllm_cmd(cfg.providers.vllm)
    return f"""PORT=$((8000 + (SLURM_JOB_ID % 2000)))
echo "Using port $PORT"
{vllm_cmd} &
VLLM_PID=$!
echo "Waiting for vLLM on port $PORT..."
until curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
    sleep 5; echo "Still waiting..."
done
echo "vLLM ready!"
nvidia-smi"""


def generate_script(cfg: Settings, config_path: Path) -> str:
    s, e, p = cfg.slurm, cfg.env, cfg.project
    experiment_name = cfg.experiment.name
    needs_vllm = cfg.experiment.mode == "vllm"
    vllm_block = (
        generate_vllm_block(cfg)
        if needs_vllm
        else 'echo "Skipping vLLM (external provider)"'
    )
    port_arg = "--port $PORT" if needs_vllm else ""
    wait_block = "kill $VLLM_PID && wait $VLLM_PID" if needs_vllm else ""
    return f"""#!/bin/bash
#SBATCH --job-name={s.job_name}_{experiment_name}
#SBATCH -p {s.partition}
#SBATCH -G {s.num_gpus}
#SBATCH -C "{s.constraint}"
#SBATCH --time={s.time}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={s.cpus}
#SBATCH --mem={s.mem}
#SBATCH --output=logs/{s.job_name}_%j.out
#SBATCH --error=logs/{s.job_name}_%j.err

export CUDA_HOME={e.cuda_home}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export HF_HOME={e.hf_home}
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE={e.hf_hub_offline}
export VLLM_BATCH_INVARIANT=1

cd {p.dir}
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

{vllm_block}

uv run {p.run_script} {config_path.as_posix()} {port_arg}

{wait_block}
"""


def submit(config_path: str, dry_run: bool = False):
    """Generate and submit a SLURM job from a yaml config.

    Args:
        config: Path to the yaml config file.
        dry_run: If True, print the script but do not submit.
    """
    config_path = Path(config_path)

    if config_path.parent == Path("."):
        config_path = CONFIGS_DIR / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = Settings.from_yaml(config_path)
    script = generate_script(cfg, config_path)

    script_path = Path(f"/tmp/{cfg.slurm.job_name}.sh")
    script_path.write_text(script)

    print("=== Generated script ===")
    print(script)

    if dry_run:
        print("=== Dry run, not submitting ===")
        return

    print(f"=== Submitting {script_path} ===")
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        cwd=Path(cfg.project.dir).parent,
    )
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        raise RuntimeError(f"sbatch error: {result.stderr.strip()}")


if __name__ == "__main__":
    fire.Fire(submit)
