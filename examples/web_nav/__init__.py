"""Web Nav: generic single-goal web navigation tasks."""

from examples import EnvSpec

SYSTEM_PROMPT = """You control a web browser through the browser_repl tool. Complete the user's task in the browser.

How to work:
1. Inspect before acting. Print `await pageInfo()` and a filtered `accessibilitySnapshot()` rather than the whole page.
2. Act with the helpers (`click`, `fillInput`, `pressKey`, `gotoUrl`, ...), then wait with `await waitForLoad()` or `await waitForElement(...)` and inspect again to confirm the result.
3. When the task is complete, reply without calling the tool. That reply is your final answer: one or two sentences saying what you did and what page the browser is on."""

EVALUATION_CRITERIA = """1. The final page shown in the screenshots must reflect the completed task (e.g. the requested article, search results, or section is open).
2. If the task asks to search, the search must have been submitted with the requested query.
3. If the task asks to open a result or item, that item's page must be open, not just the list containing it.
4. The agent's final answer must be consistent with what the screenshots show."""

WEB_NAV = EnvSpec(
    name="web_nav",
    system_prompt=SYSTEM_PROMPT,
    evaluation_criteria=EVALUATION_CRITERIA,
    task_file="examples/web_nav/tasks.jsonl",
)
