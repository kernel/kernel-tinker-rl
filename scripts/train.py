#!/usr/bin/env python3
"""
train.py - GRPO training for browser agents that use the Kernel Browser REPL.

Each step samples `batch_size` tasks and runs `group_size` rollouts per task, each in
its own browser from the pool, so the pool needs batch_size * group_size browsers.

Usage:
    uv run python -m scripts.train env=agent_auth pool_name=rl-training

    # With a held-out eval set and W&B logging
    uv run python -m scripts.train \
        env=agent_auth \
        task_file=examples/agent_auth/tasks_train.jsonl \
        eval_task_file=examples/agent_auth/tasks_eval.jsonl \
        pool_name=rl-training \
        wandb_project=browser-repl-rl

Output goes to results/<run_name>/ (metrics.jsonl, checkpoints.jsonl, rollout logs).

Environment variables:
    KERNEL_API_KEY, TINKER_API_KEY, OPENROUTER_API_KEY (WebJudge), WANDB_API_KEY (optional)
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime

import chz
from dotenv import load_dotenv
from tinker_cookbook import cli_utils, hyperparam_utils
from tinker_cookbook.rl import train

from core.env import BrowserReplDatasetBuilder
from examples import get_env

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@chz.chz
class CLIConfig:
    # Environment
    env: str = "agent_auth"
    task_file: str | None = None  # defaults to the env's tasks.jsonl
    eval_task_file: str | None = None
    max_tasks: int | None = None
    pool_name: str = "rl-training"

    # Model
    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    renderer_name: str = "qwen3_5_disable_thinking"
    lora_rank: int = 32
    learning_rate: float | None = None  # defaults to hyperparam_utils.get_lr(model_name)

    # Rollouts
    batch_size: int = 8
    group_size: int = 8
    max_tokens: int = 1024
    max_turns: int = 10
    max_trajectory_tokens: int = 32 * 1024
    temperature: float = 1.0
    webjudge_model: str = "openai/gpt-5-mini"
    seed: int = 0

    # Logging and checkpoints
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    eval_every: int = 10
    save_every: int = 10
    load_checkpoint_path: str | None = None
    max_steps: int | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cfg: CLIConfig) -> None:
    env = get_env(cfg.env)
    learning_rate = cfg.learning_rate or hyperparam_utils.get_lr(cfg.model_name)

    builder = BrowserReplDatasetBuilder(
        env_name=env.name,
        system_prompt=env.system_prompt,
        evaluation_criteria=env.evaluation_criteria,
        model_name=cfg.model_name,
        renderer_name=cfg.renderer_name,
        task_file=cfg.task_file or env.task_file,
        eval_task_file=cfg.eval_task_file,
        max_tasks=cfg.max_tasks,
        pool_name=cfg.pool_name,
        batch_size=cfg.batch_size,
        group_size=cfg.group_size,
        max_turns=cfg.max_turns,
        max_trajectory_tokens=cfg.max_trajectory_tokens,
        webjudge_model=cfg.webjudge_model,
        seed=cfg.seed,
    )

    model_short = cfg.model_name.split("/")[-1].lower()
    run_name = (
        f"{env.name}_{model_short}_bs{cfg.batch_size}_gs{cfg.group_size}_"
        f"lr{learning_rate:.1e}_rank{cfg.lora_rank}_{datetime.now():%Y-%m-%d-%H-%M}"
    )
    log_path = cfg.log_path or f"results/{run_name}"
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)

    config = train.Config(
        model_name=cfg.model_name,
        recipe_name="kernel_browser_repl",
        renderer_name=cfg.renderer_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=learning_rate,
        lora_rank=cfg.lora_rank,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        eval_every=cfg.eval_every,
        save_every=cfg.save_every,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name or run_name,
        load_checkpoint_path=cfg.load_checkpoint_path,
        max_steps=cfg.max_steps,
        termination=builder.rollout_config().termination,
        # Real websites are flaky; retry failed rollouts instead of crashing the run.
        rollout_error_tolerance=True,
    )
    await train.main(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
