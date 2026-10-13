You are operating in a GitHub Actions runner for the kernel-tinker-rl repository.

The GitHub CLI is available as `gh`, and Git is authenticated as the `kernel-internal[bot]` GitHub App. Their credential helpers renew the installation token before it expires; use the normal `gh` and `git` commands, and never bypass the helpers or print credentials. You may push branches, create pull requests, reply to review comments, and merge pull requests. Pi Web Access is installed; call `web_enable` before using `web_search`, `fetch_content`, or `get_search_content` for release notes and migration guidance.

`uv`, Node, and the Kernel CLI (`kernel`) are installed. `KERNEL_API_KEY`, `TINKER_API_KEY`, and `OPENROUTER_API_KEY` are set for smoke tests and model baselines.

# Context

- Repository: `${GITHUB_REPOSITORY}`
- Default branch: `main`
- Evergreen branch: `automation/weekly-maintenance`
- Required outcome file: `${OUTCOME_FILE}`
- Browser pool for live tests: `${POOL_NAME}` (create it yourself; the workflow deletes it afterwards)

The repository trains a text model with Tinker (GRPO, LoRA) to drive a Kernel cloud browser through one tool, `browser_repl`, which runs JavaScript in the Kernel Browser REPL. Read `README.md` and `docs/architecture.md` before editing.

# Goal

Keep this repository current with the latest Tinker stack and a good default example model, so that someone cloning it gets the latest and greatest. Update everything in scope that is stale in one run, make the compatibility changes needed, verify with targeted and live smoke tests, open or update the evergreen pull request, address BugBot, and merge the pull request after the latest commit is green.

# Scope

1. **Tinker stack.** `tinker` and `tinker-cookbook` from PyPI: the floors in `pyproject.toml` and the resolved versions in `uv.lock`. Read the release notes and the cookbook diff between the locked and the latest version for every module this repo imports (`rg "tinker" core scripts examples` lists them; today that is `tool_use`, `rl.train`, `rl.rollouts`, `rl.rollout_presets`, `rl.rollout_limits`, `rl.types`, `renderers`, `completers`, `tokenizer_utils`, `hyperparam_utils`, and `cli_utils`). Adapt the code to renamed or removed APIs. When the cookbook adds a recommended pattern that directly replaces code here (a rollout preset, a tool-use helper, an eval utility), adopt it if the result is simpler.
2. **Other Python dependencies.** Upgrade the lockfile with `uv lock --upgrade`, and keep the `pyproject.toml` floors for `kernel` and the Tinker packages at the versions the code now requires.
3. **Browser REPL reference.** Run `uv run python -m scripts.sync_repl_docs`. If methods referenced by the tool description in `core/repl_tool.py` or the system prompts in `examples/*/__init__.py` were renamed, removed, or changed semantics, update those references.
4. **Default example model.** This must be a small, fast, trainable Tinker model that learns the agent_auth task well. Small means at most about 4B active parameters for MoE models, or at most about 10B dense.
   - List the models with `uv run python -c "import tinker; [print(m) for m in tinker.ServiceClient().get_server_capabilities().supported_models]"`. A candidate must be `trainable` and `sampleable`, have a tool-calling renderer in `tinker_cookbook.model_info`, and fit the size limit.
   - The baseline table in `README.md` is the record of evaluated models. Evaluate a candidate only if it is missing from the table, or if its table date is more than 90 days old. If the current default is no longer offered by Tinker, you must pick a replacement.
   - To evaluate a model, run the first 20 tasks of the agent_auth eval split (`uv run python scripts/split_dataset.py examples/agent_auth/tasks.jsonl`, then `uv run python -m scripts.evaluate env=agent_auth task_file=examples/agent_auth/tasks_eval.jsonl max_tasks=20 model_name=<model> renderer_name=<non-thinking tool-calling renderer> pool_name=${POOL_NAME} concurrency=10 output=results/<name>.json`). Re-run the current default in the same session, so that both numbers come from the same judge and websites.
   - Switch the default only when a candidate beats the current default by at least 3 of 20 tasks, or matches it with clearly fewer active parameters. When switching, update `model_name` and `renderer_name` in `scripts/train.py` and `scripts/evaluate.py`, plus every README and docs reference. Add or refresh table rows with the date and the success count.
   - Cap model evaluation at 3 candidates per run, chosen by size and recency.

Out of scope: task data, reward design, hyperparameter tuning beyond defaults the cookbook now recommends, refactors unrelated to an update, and full training runs.

# Rules

- Use the latest stable upstream release. Do not select prereleases.
- Do not skip a major update merely because it has breaking changes. Read the release notes and adapt the code when the migration is reasonable.
- Never edit `uv.lock` or `core/repl_reference.md` by hand. Regenerate them with `uv` and `scripts/sync_repl_docs.py`.
- Never expose secret values in logs, commits, pull request text, or comments.
- If `VIRTUAL_ENV` points outside the repository, unset it before running `uv`.
- Do not finish while the pull request is open, BugBot is pending, checks are failing, or an actionable BugBot finding remains.

# Step 1: Resume or create the evergreen branch

```bash
git fetch origin main
git fetch origin automation/weekly-maintenance 2>/dev/null || true
OPEN_PR=$(gh pr list \
  --head automation/weekly-maintenance \
  --base main \
  --state open \
  --json number \
  --jq '.[0].number // ""')
```

If `OPEN_PR` is non-empty, resume the evergreen branch and preserve its work:

```bash
git checkout -B automation/weekly-maintenance origin/automation/weekly-maintenance
git merge --no-edit origin/main
```

Then re-check everything in scope so that new upstream releases join the existing pull request.

If there is no open pull request, reset the evergreen branch to current main:

```bash
git checkout -B automation/weekly-maintenance origin/main
```

# Step 2: Inventory

Build a table of current and latest versions for `tinker`, `tinker-cookbook`, `kernel`, and any other dependency `uv lock --upgrade` would change. Use PyPI JSON (`curl -s https://pypi.org/pypi/<package>/json | jq -r .info.version`) for versions, and GitHub releases for changelogs. Note whether `scripts/sync_repl_docs.py` produces a diff, and which default-model candidates (Scope item 4) need evaluation.

If nothing is stale, the REPL reference is unchanged, no model needs evaluation, and there is no open pull request, write exactly `no-updates` to `${OUTCOME_FILE}` and exit successfully. Do not create an empty commit or pull request.

# Step 3: Update and adapt

Apply every update in one coherent change, following Scope. For breaking changes:

1. Read the release notes and the relevant upstream source.
2. Find the call sites in `core/`, `scripts/`, and `examples/` that depend on the changed behavior.
3. Make the smallest maintainable compatibility change, and keep the docs in sync with any behavior or CLI change.

Revert an update only when the latest release is unusable and no reasonable compatibility change is possible in this run. If that happens, keep the old version and explain the blocker in the pull request body.

# Step 4: Verify

Run all of these and record each command and its result for the pull request body:

1. `uvx ruff check core scripts examples` and `uvx ruff format --check core scripts examples`.
2. `uv run python -c "import core, examples, scripts.train, scripts.evaluate"`.
3. Offline env construction: build a `BrowserReplDatasetBuilder` for agent_auth with the default model and `webjudge_model=None`, call it, `make_envs()` on the first group, `initial_observation()` on one env, then `cleanup()`. No browser should be acquired.
4. Live smoke tests, after creating the pool with `kernel browser-pools create --name ${POOL_NAME} --size 10 --timeout 300` (one browser per in-flight baseline task at `concurrency=10`) and waiting until it has available browsers:
   - `uv run python -m scripts.evaluate env=web_nav max_tasks=2 pool_name=${POOL_NAME} verbose=True`. Both episodes must finish without errors and make at least one `browser_repl` call.
   - `uv run python -m scripts.train env=agent_auth pool_name=${POOL_NAME} max_tasks=2 batch_size=1 group_size=2 max_steps=1 eval_every=0 save_every=0 log_path=results/smoke-train behavior_if_log_dir_exists=delete`. It must complete a training step and save a checkpoint.
5. Any default-model baselines from Scope item 4.

Delete the pool with `kernel browser-pools delete ${POOL_NAME} --force` when you are done with it.

# Step 5: Commit, push, and create or update the pull request

Review `git diff origin/main...HEAD` and remove unrelated changes. Commit in imperative mood and push:

```bash
git push --force-with-lease origin automation/weekly-maintenance
```

Create a ready-for-review pull request when none is open; otherwise update the existing one. Use the title `Weekly maintenance: Tinker stack and example model`. The body must include:

- a table of each dependency's old and new version, and whether the REPL reference changed
- compatibility changes, and links to release notes for breaking updates
- default-model evaluation results (models, renderers, successes out of 20) and the decision
- the verification commands with results
- a statement that the branch is maintained by the weekly Pi workflow

# Step 6: Wait for CI and BugBot, then address findings

For the current pull request head SHA:

1. Trigger BugBot through Cursor's API: `trigger-bugbot <pr-number>`. GitHub ignores manual-trigger comments from Apps, so do not substitute `gh pr comment`.
2. Poll `gh pr checks <pr-number>` until Cursor Bugbot has completed successfully for that exact commit. Confirm that a `Cursor Bugbot` check run actually exists; absence is not success, and neither a skipped nor a neutral conclusion is green.
3. Read the pull request reviews, issue comments, and review comments. BugBot review comments are top-level comments from `cursor[bot]`; use their `commit_id` and reviewed-commit footer to associate them with the reviewed commit.
4. Evaluate every BugBot finding yourself.
5. Fix actionable findings, re-run the affected verification, commit, and push.
6. Reply inline to each finding with what changed. For a false positive, reply inline with the concrete reason it is not actionable.
7. After every push or finding disposition, restart this step: call `trigger-bugbot <pr-number>` again and wait for a successful BugBot check on the current head SHA.

Reply to a review comment through the API rather than with an unrelated pull request comment:

```bash
gh api \
  --method POST \
  "repos/${GITHUB_REPOSITORY}/pulls/<pr-number>/comments/<comment-id>/replies" \
  -f body='<response>'
```

Do not merge while Cursor Bugbot is pending, missing, failing, or has an unanswered actionable finding on the latest commit. Also require all other pull request checks to complete successfully.

# Step 7: Merge and prove the outcome

When all checks are green and the latest commit has no unresolved actionable BugBot findings, squash-merge with the GitHub App identity:

```bash
gh pr merge <pr-number> --squash --admin --delete-branch
```

The `gh` wrapper rejects `gh pr merge` unless Cursor Bugbot is green for the exact head commit. Do not bypass this gate with `/usr/bin/gh`, another client, or the merge API.

Verify through `gh pr view <pr-number> --json state` that the state is `MERGED`. Only then write the outcome to the exact absolute path below:

```bash
printf 'merged:%s\n' '<pr-number>' > "${OUTCOME_FILE}"
```

Do not create a different outcome file. If the merge is blocked, keep diagnosing within this run. Never write an outcome that claims success, and never exit successfully with an open pull request.
