#!/usr/bin/env python3
"""
evaluate.py - Run the agent on tasks with Tinker sampling and grade with WebJudge.

Works for the base model and for trained checkpoints, on a task file or one ad-hoc task.

Usage:
    # One ad-hoc task, printing every REPL cell
    uv run python -m scripts.evaluate \
        url=https://github.com task="Find the sign in page and list its input fields" \
        verbose=True

    # Baseline on the eval split
    uv run python -m scripts.evaluate \
        env=agent_auth task_file=examples/agent_auth/tasks_eval.jsonl \
        output=results/baseline_eval.json

    # A trained checkpoint (sampler weights path from checkpoints.jsonl)
    uv run python -m scripts.evaluate \
        env=agent_auth task_file=examples/agent_auth/tasks_eval.jsonl \
        model_path=tinker://<run-id>/sampler_weights/<name> \
        output=results/trained_eval.json

Environment variables:
    KERNEL_API_KEY, TINKER_API_KEY, OPENROUTER_API_KEY (WebJudge)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import chz
import tinker
from dotenv import load_dotenv
from rich.console import Console
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.rollouts import do_group_rollout

from core.env import (
    BrowserReplDatasetBuilder,
    BrowserReplEnvGroupBuilder,
    BrowserTask,
    episode_summaries,
    load_tasks,
)
from examples import get_env

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logger = logging.getLogger(__name__)
console = Console()


@chz.chz
class CLIConfig:
    env: str = "agent_auth"
    task_file: str | None = None  # defaults to the env's tasks.jsonl
    max_tasks: int | None = None
    url: str | None = None  # with `task`, run one ad-hoc task instead of a task file
    task: str | None = None
    pool_name: str = "rl-training"

    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    model_path: str | None = None  # tinker:// sampler weights; None evaluates the base model
    renderer_name: str = "qwen3_5_disable_thinking"
    max_tokens: int = 1024
    temperature: float = 1.0
    max_turns: int = 10
    group_size: int = 1  # rollouts per task
    concurrency: int = 8  # tasks in flight; needs concurrency * group_size browsers
    webjudge_model: str | None = "openai/gpt-5-mini"

    output: str | None = None
    verbose: bool = False


def print_episode(episode: dict[str, Any]) -> None:
    for i, call in enumerate(episode["repl_calls"], 1):
        console.rule(f"[dim]{episode['task_id']} · call {i}")
        console.print(call["code"], markup=False, highlight=False)
        console.print(f"[dim]→ {call['output'][:1500]}[/]", markup=True, highlight=False)
    console.rule(f"[dim]{episode['task_id']} · final answer")
    console.print(episode["final_answer"] or "(empty)", markup=False, highlight=False)
    if episode["judge_reasoning"]:
        console.print(f"[dim]{episode['judge_reasoning']}[/]", highlight=False)


async def run_task(
    builder: BrowserReplDatasetBuilder,
    task: BrowserTask,
    cfg: CLIConfig,
    policy: TinkerTokenCompleter,
) -> list[dict[str, Any]]:
    group = builder.build_env_group_builder(task, cfg.group_size)
    assert isinstance(group, BrowserReplEnvGroupBuilder)
    try:
        result = await do_group_rollout(
            group, policy, termination=builder.rollout_config().termination
        )
    except Exception as e:
        logger.exception("Rollout failed for %s", task.id)
        return [{"task_id": task.id, "task": task.task, "reward": 0.0, "error": repr(e)}]

    episodes = episode_summaries(group, result)
    for episode in episodes:
        mark = "[green]✓[/]" if episode["judge_success"] else "[red]✗[/]"
        console.print(
            f"{mark} {task.id}: reward={episode['reward']:.2f} "
            f"repl_calls={len(episode['repl_calls'])} stop={episode['stop_reason']}"
        )
        if cfg.verbose:
            print_episode(episode)
    return episodes


async def main(cfg: CLIConfig) -> None:
    env = get_env(cfg.env)
    if cfg.url and cfg.task:
        tasks = [BrowserTask(id="adhoc", initial_url=cfg.url, task=cfg.task)]
    else:
        tasks = load_tasks(cfg.task_file or env.task_file, limit=cfg.max_tasks)

    builder = BrowserReplDatasetBuilder(
        env_name=env.name,
        system_prompt=env.system_prompt,
        evaluation_criteria=env.evaluation_criteria,
        model_name=cfg.model_name,
        renderer_name=cfg.renderer_name,
        task_file=cfg.task_file or env.task_file,
        pool_name=cfg.pool_name,
        batch_size=1,
        group_size=cfg.group_size,
        max_turns=cfg.max_turns,
        webjudge_model=cfg.webjudge_model,
    )

    service = tinker.ServiceClient()
    if cfg.model_path:
        sampling_client = await service.create_sampling_client_async(model_path=cfg.model_path)
    else:
        sampling_client = await service.create_sampling_client_async(base_model=cfg.model_name)
    policy = TinkerTokenCompleter(
        sampling_client, max_tokens=cfg.max_tokens, temperature=cfg.temperature
    )

    console.print(
        f"Evaluating {cfg.model_path or cfg.model_name} on {len(tasks)} task(s) "
        f"x {cfg.group_size} rollout(s)"
    )
    semaphore = asyncio.Semaphore(cfg.concurrency)

    async def bounded(task: BrowserTask) -> list[dict[str, Any]]:
        async with semaphore:
            return await run_task(builder, task, cfg, policy)

    groups = await asyncio.gather(*[bounded(task) for task in tasks])
    episodes = [episode for group in groups for episode in group]

    # Count rewarded episodes, as training does: WebJudge can pass an episode that hit
    # max_turns without a final answer, but the termination policy zeroes its reward.
    n = max(len(episodes), 1)
    successes = sum(e["reward"] > 0 for e in episodes)
    judge_successes = sum(bool(e.get("judge_success")) for e in episodes)
    console.print(
        f"\nSuccess rate: {successes}/{len(episodes)} ({successes / n:.1%}), "
        f"judge success: {judge_successes}/{len(episodes)} ({judge_successes / n:.1%}), "
        f"errors: {sum('error' in e for e in episodes)}"
    )

    if cfg.output:
        Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "success_rate": successes / n,
            "judge_success_rate": judge_successes / n,
            "episodes": len(episodes),
        }
        with open(cfg.output, "w") as f:
            json.dump(
                {"config": chz.asdict(cfg), "summary": summary, "results": episodes}, f, indent=2
            )
        console.print(f"Results written to {cfg.output}")


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main(chz.entrypoint(CLIConfig)))
