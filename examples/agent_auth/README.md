# Agent Auth - Login Discovery

Train an agent to find a site's login or registration page and report the input fields needed to start that flow.

## How an episode works

1. The browser opens at the site's homepage on the agent's first `browser_repl` call.
2. The agent finds the sign-in or sign-up entry point in an accessibility snapshot, clicks through, and lists the visible inputs.
3. It replies without a tool call; that reply is the field list, e.g. `- Email address (email)`.
4. WebJudge checks that the final screenshots show an authentication page and that the reported fields match what is visible. Progressive disclosure (email first, password later) counts as success.

The system prompt, including a worked example, and the judge criteria are in [`__init__.py`](./__init__.py).

## Task data

`tasks.jsonl` holds 467 LLM-generated tasks across the top ~1000 sites, e.g.:

```json
{"id": "github-com-sign-in", "initial_url": "https://github.com", "task": "Navigate to github.com and find the sign in page. ..."}
```

## Running

```bash
uv run python -m scripts.evaluate env=agent_auth max_tasks=5 pool_name=rl-training verbose=True
uv run python -m scripts.train env=agent_auth pool_name=rl-training
```
