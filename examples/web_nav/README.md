# Web Nav

Generic single-goal navigation tasks (open an article, run a search, click the top story). It's useful as a smoke test, or as a starting point for your own tasks.

## Usage

```bash
# All tasks in tasks.jsonl
uv run python -m scripts.evaluate env=web_nav pool_name=rl-training verbose=True

# An ad-hoc task with this env's prompt and judge criteria
uv run python -m scripts.evaluate env=web_nav pool_name=rl-training verbose=True \
    url=https://example.com task="Click on the first link"
```

## Adding tasks

```jsonl
{"id": "unique-id", "initial_url": "https://example.com", "task": "Description of what to do"}
```
