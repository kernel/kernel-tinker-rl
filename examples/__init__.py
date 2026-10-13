"""Task environments: a system prompt, WebJudge criteria, and a task file per use case."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvSpec:
    name: str
    system_prompt: str
    evaluation_criteria: str
    task_file: str


def get_env(name: str) -> EnvSpec:
    from examples.agent_auth import AGENT_AUTH
    from examples.web_nav import WEB_NAV

    envs = {env.name: env for env in (AGENT_AUTH, WEB_NAV)}
    if name not in envs:
        raise ValueError(f"Unknown env {name!r}; choose from {sorted(envs)}")
    return envs[name]
