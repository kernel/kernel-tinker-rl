# Custom Environments

An environment here is three things: a system prompt, WebJudge evaluation criteria, and a task file. The browser tool, rollout loop, and reward plumbing are shared.

## 1. Write tasks

`examples/form_filling/tasks.jsonl`:

```json
{"id": "contact-form", "initial_url": "https://example.com/contact", "task": "Fill out the contact form with name 'Jane Doe' and email 'jane@example.com', but do not submit it."}
```

Each line needs `id`, `initial_url`, and `task`. The browser opens at `initial_url` on the agent's first tool call.

## 2. Define the environment

`examples/form_filling/__init__.py`:

```python
from examples import EnvSpec

SYSTEM_PROMPT = """You control a web browser through the browser_repl tool. Fill in the form the user describes.

How to work:
1. Inspect the form with a filtered accessibilitySnapshot() (textbox, combobox, checkbox nodes).
2. Fill each field with `await fillInput(node, value)` and confirm the values with js().
3. Do not submit the form.
4. When done, reply without calling the tool, listing each field and the value you entered."""

EVALUATION_CRITERIA = """1. Every requested field must show the requested value in the final screenshot.
2. The form must not have been submitted.
3. The final answer must match the values visible on the page."""

FORM_FILLING = EnvSpec(
    name="form_filling",
    system_prompt=SYSTEM_PROMPT,
    evaluation_criteria=EVALUATION_CRITERIA,
    task_file="examples/form_filling/tasks.jsonl",
)
```

The prompt should say how the episode ends (reply without a tool call) and what the final answer looks like, because WebJudge checks the answer against the screenshots.

## 3. Register it

Add it to `get_env` in `examples/__init__.py`:

```python
from examples.form_filling import FORM_FILLING

envs = {env.name: env for env in (AGENT_AUTH, WEB_NAV, FORM_FILLING)}
```

## 4. Try it, then train

```bash
uv run python -m scripts.evaluate env=form_filling pool_name=rl-training verbose=True max_tasks=3
uv run python -m scripts.train env=form_filling pool_name=rl-training
```

## Tips

- **Check the baseline first.** GRPO learns only from groups with mixed outcomes. If the base model never succeeds, make tasks easier, raise `group_size`, or start from a stronger model. If it always succeeds, make them harder.
- **Read transcripts.** `verbose=True` shows every cell. Recurring mistakes (no `repl.write`, oversized snapshots, giving up early) are what training should remove. If the prompt can fix one cheaply, fix it there.
- **Keep criteria checkable from screenshots.** WebJudge sees the screenshots, the code the agent ran, and the final answer. It cannot see DOM state that is not rendered.
- **Different reward?** Pass your own `reward_fn` to `build_agent_tool_env` in `BrowserReplEnvGroupBuilder.make_envs()`. It receives the episode's messages and returns `(reward, metrics)`; `BrowserReplTool.calls` holds every cell and screenshot if you need browser-side evidence.
