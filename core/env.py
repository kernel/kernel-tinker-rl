"""
Tinker RL environment for browser tasks solved with the browser_repl tool.

Each rollout gets its own browser from a Kernel browser pool, opened at the task's initial
URL on the first browser_repl call. The model calls browser_repl until it replies without
a tool call; that reply is its final answer. WebJudge grades the episode from the REPL
cells, the final answer, and screenshots captured after each cell.
"""

from __future__ import annotations

import io
import json
import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import chz
from kernel import AsyncKernel, KernelError
from PIL import Image
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.renderers.base import Message, get_text_content
from tinker_cookbook.rl.rollout_limits import RolloutLimits
from tinker_cookbook.rl.rollout_presets import RolloutConfig, agentic
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.tool_use import build_agent_tool_env

from core.repl_tool import BrowserReplTool, ReplCall
from core.reward_models import Trajectory, WebJudge

logger = logging.getLogger(__name__)

JUDGE_SCREENSHOT_MAX_SIZE = 1280
JUDGE_CODE_MAX_CHARS = 1000

# Cookbook envs render prompts in worker threads. Transformers 5 mutates the shared
# fast-tokenizer backend during encode(), so concurrent calls can fail with
# `RuntimeError: Already borrowed`. Serialize tokenizer operations within each process.
_TOKENIZER_LOCK = RLock()


class _SynchronizedTokenizer:
    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._tokenizer, name)
        if not callable(attribute):
            return attribute

        def synchronized(*args: Any, **kwargs: Any) -> Any:
            with _TOKENIZER_LOCK:
                return attribute(*args, **kwargs)

        return synchronized

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with _TOKENIZER_LOCK:
            return self._tokenizer(*args, **kwargs)


def _synchronized_tokenizer(tokenizer: Any) -> _SynchronizedTokenizer:
    if isinstance(tokenizer, _SynchronizedTokenizer):
        return tokenizer
    return _SynchronizedTokenizer(tokenizer)


@dataclass(frozen=True)
class BrowserTask:
    id: str
    initial_url: str
    task: str


def load_tasks(path: str | Path, limit: int | None = None) -> list[BrowserTask]:
    """Load tasks from a JSONL file with id, initial_url, and task fields."""
    tasks: list[BrowserTask] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            tasks.append(BrowserTask(data["id"], data["initial_url"], data["task"]))
            if limit is not None and len(tasks) >= limit:
                break
    return tasks


def make_rollout_config(max_turns: int, max_trajectory_tokens: int) -> RolloutConfig:
    """The cookbook's agentic preset (sequential tool calls, parse-error retries,
    no positive reward for limit-stopped episodes) with our turn and token budgets."""
    return chz.replace(
        agentic(),
        limits=RolloutLimits(
            max_turns=max_turns,
            max_trajectory_tokens=max_trajectory_tokens,
            max_tool_calls=max_turns,
        ),
    )


def _to_image(png: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(png))
    image.thumbnail((JUDGE_SCREENSHOT_MAX_SIZE, JUDGE_SCREENSHOT_MAX_SIZE))
    return image


@dataclass
class EpisodeResult:
    """What happened in one episode, filled in when it is graded."""

    final_answer: str = ""
    calls: list[ReplCall] = field(default_factory=list)
    judge_success: bool | None = None
    judge_reasoning: str = ""


class WebJudgeReward:
    """Episode-end reward: WebJudge success (1.0) or failure (0.0)."""

    def __init__(self, task: BrowserTask, tool: BrowserReplTool, webjudge: WebJudge | None):
        self.task = task
        self.tool = tool
        self.webjudge = webjudge
        self.result = EpisodeResult()

    async def __call__(self, messages: list[Message]) -> tuple[float, dict[str, float]]:
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        final_answer = get_text_content(assistant_messages[-1]) if assistant_messages else ""
        calls = self.tool.calls
        self.result = EpisodeResult(final_answer=final_answer, calls=list(calls))

        metrics: dict[str, float] = {
            "repl_calls": len(calls),
            "repl_errors": sum(c.failed for c in calls),
        }
        # Every task requires looking at the browser; an answer without it earns nothing.
        if self.webjudge is None or not calls:
            return 0.0, metrics

        screenshots = [c.screenshot for c in calls if c.screenshot]
        action_history = [f"browser_repl:\n{c.code[:JUDGE_CODE_MAX_CHARS]}" for c in calls]
        action_history.append(f"Final answer: {final_answer}")

        try:
            judged = await self.webjudge.evaluate(
                Trajectory(
                    task_id=self.task.id,
                    task=self.task.task,
                    action_history=action_history,
                    screenshots=[_to_image(s) for s in screenshots],
                )
            )
        except Exception as e:
            logger.warning("WebJudge failed for %s: %s", self.task.id, e)
            return 0.0, {**metrics, "webjudge_error": 1.0}

        self.result.judge_success = judged.success
        self.result.judge_reasoning = judged.reasoning
        return judged.score, {**metrics, "webjudge_success": float(judged.success)}


class BrowserReplEnvGroupBuilder(EnvGroupBuilder):
    """`group_size` rollouts of one task, each in its own pooled browser."""

    def __init__(
        self,
        task: BrowserTask,
        group_size: int,
        kernel: AsyncKernel,
        pool_name: str,
        renderer: Renderer,
        system_prompt: str,
        webjudge: WebJudge | None,
        rollout_config: RolloutConfig,
        env_name: str,
        acquire_timeout_seconds: int,
        repl_timeout_sec: int,
    ):
        self.task = task
        self.group_size = group_size
        self.kernel = kernel
        self.pool_name = pool_name
        self.renderer = renderer
        self.system_prompt = system_prompt
        self.webjudge = webjudge
        self.rollout_config = rollout_config
        self.env_name = env_name
        self.acquire_timeout_seconds = acquire_timeout_seconds
        self.repl_timeout_sec = repl_timeout_sec
        self.tools: list[BrowserReplTool] = []
        self.rewards: list[WebJudgeReward] = []

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore tokenizer synchronization after the renderer is unpickled."""
        self.__dict__.update(state)
        self.renderer.tokenizer = _synchronized_tokenizer(self.renderer.tokenizer)

    async def make_envs(self) -> Sequence[Env]:
        # Retry strategies call make_envs() again before cleanup(), so accumulate tools
        # and rewards rather than replacing them; cleanup() releases every browser.
        tools = [
            BrowserReplTool(
                self.kernel,
                self.pool_name,
                self.task.initial_url,
                acquire_timeout_seconds=self.acquire_timeout_seconds,
                timeout_sec=self.repl_timeout_sec,
            )
            for _ in range(self.group_size)
        ]
        self.tools.extend(tools)

        prefix = self.renderer.create_conversation_prefix_with_tools(
            tools=[tools[0].to_spec()], system_prompt=self.system_prompt
        )
        initial_messages = prefix + [
            {
                "role": "user",
                "content": f"{self.task.task}\n\nThe browser is open at {self.task.initial_url}.",
            }
        ]

        envs = []
        for tool in tools:
            reward = WebJudgeReward(self.task, tool, self.webjudge)
            self.rewards.append(reward)
            envs.append(
                build_agent_tool_env(
                    renderer=self.renderer,
                    tools=[tool],
                    initial_messages=initial_messages,
                    reward_fn=reward,
                    rollout_config=self.rollout_config,
                )
            )
        return envs

    async def cleanup(self) -> None:
        for tool in self.tools:
            try:
                await tool.release()
            except KernelError as e:
                logger.warning("Failed to release browser for %s: %s", self.task.id, e)

    def logging_tags(self) -> list[str]:
        return [self.env_name]


class BrowserTaskDataset(RLDataset):
    def __init__(
        self,
        tasks: list[BrowserTask],
        batch_size: int,
        make_builder: Callable[[BrowserTask], EnvGroupBuilder],
    ):
        self.tasks = tasks
        self.batch_size = batch_size
        self.make_builder = make_builder

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch = self.tasks[index * self.batch_size : (index + 1) * self.batch_size]
        return [self.make_builder(task) for task in batch]

    def __len__(self) -> int:
        return (len(self.tasks) + self.batch_size - 1) // self.batch_size


@chz.chz
class BrowserReplDatasetBuilder(RLDatasetBuilder):
    """Builds train (and optionally eval) datasets of browser_repl tasks."""

    env_name: str
    system_prompt: str
    evaluation_criteria: str
    model_name: str
    renderer_name: str
    task_file: str
    pool_name: str
    batch_size: int
    group_size: int
    eval_task_file: str | None = None
    eval_group_size: int = 1
    max_tasks: int | None = None
    num_epochs: int = 1
    seed: int = 0
    max_turns: int = 10
    max_trajectory_tokens: int = 32 * 1024
    # Long enough to queue for a browser when more rollouts run than the pool holds
    # (e.g. in-training eval over more tasks than the pool size).
    acquire_timeout_seconds: int = 600
    repl_timeout_sec: int = 30
    webjudge_model: str | None = "openai/gpt-5-mini"

    def build_env_group_builder(self, task: BrowserTask, group_size: int) -> EnvGroupBuilder:
        return BrowserReplEnvGroupBuilder(
            task=task,
            group_size=group_size,
            kernel=self._kernel,
            pool_name=self.pool_name,
            renderer=self._renderer,
            system_prompt=self.system_prompt,
            webjudge=self._webjudge,
            rollout_config=self.rollout_config(),
            env_name=self.env_name,
            acquire_timeout_seconds=self.acquire_timeout_seconds,
            repl_timeout_sec=self.repl_timeout_sec,
        )

    def rollout_config(self) -> RolloutConfig:
        return make_rollout_config(self.max_turns, self.max_trajectory_tokens)

    @chz.init_property
    def _kernel(self) -> AsyncKernel:
        return AsyncKernel()

    @chz.init_property
    def _renderer(self) -> Renderer:
        tokenizer = _synchronized_tokenizer(tokenizer_utils.get_tokenizer(self.model_name))
        return get_renderer(self.renderer_name, tokenizer, model_name=self.model_name)

    @chz.init_property
    def _webjudge(self) -> WebJudge | None:
        if self.webjudge_model is None:
            return None
        return WebJudge(model=self.webjudge_model, evaluation_criteria=self.evaluation_criteria)

    def _dataset(
        self, task_file: str, group_size: int, shuffle: bool, num_epochs: int = 1
    ) -> BrowserTaskDataset:
        loaded = load_tasks(task_file, limit=self.max_tasks)
        # The trainer stops after one pass over the dataset, so repeat it once per epoch,
        # reshuffled each time.
        tasks: list[BrowserTask] = []
        for epoch in range(num_epochs):
            epoch_tasks = list(loaded)
            if shuffle:
                random.Random(self.seed + epoch).shuffle(epoch_tasks)
            tasks.extend(epoch_tasks)
        return BrowserTaskDataset(
            tasks, self.batch_size, lambda task: self.build_env_group_builder(task, group_size)
        )

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        train = self._dataset(
            self.task_file, self.group_size, shuffle=True, num_epochs=self.num_epochs
        )
        test = None
        if self.eval_task_file is not None:
            test = self._dataset(self.eval_task_file, self.eval_group_size, shuffle=False)
        return train, test


def episode_summaries(
    builder: BrowserReplEnvGroupBuilder, group: TrajectoryGroup
) -> list[dict[str, Any]]:
    """Per-rollout details for a group finished by do_group_rollout."""
    return [
        {
            "task_id": builder.task.id,
            "task": builder.task.task,
            "initial_url": builder.task.initial_url,
            "reward": reward,
            "stop_reason": trajectory.stop_reason,
            "judge_success": r.result.judge_success,
            "final_answer": r.result.final_answer,
            "judge_reasoning": r.result.judge_reasoning,
            "repl_calls": [{"code": c.code, "output": c.output} for c in r.result.calls],
        }
        for r, reward, trajectory in zip(
            builder.rewards, group.get_total_rewards(), group.trajectories_G, strict=True
        )
    ]
