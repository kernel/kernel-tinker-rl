"""Agent Auth: find a site's login or registration page and report its input fields."""

from examples import EnvSpec

SYSTEM_PROMPT = """You control a web browser through the browser_repl tool. The user wants you to find a login or registration page and report the input fields on it.

How to work:
1. If a cookie banner, modal, or popup covers the page, dismiss it first.
2. Find links or buttons for the page you need in a filtered `accessibilitySnapshot()`, and print only those nodes.
3. Click the matching node with `await click({backendNodeId})`, then `await waitForLoad()` and inspect again. If the click fails (intercepted, not visible, outside viewport), get the link's URL with `js()` and open it with `await gotoUrl(url)`.
4. On the authentication page, print the URL and the input fields (textbox, combobox, and checkbox nodes) that are visible. The accessibility snapshot also includes hidden fields, such as a password field that appears only after the email step, so check each one with `await waitForElement({backendNodeId}, {state: "visible", timeoutSec: 1})` and report only those that return true.
5. Do not type into or submit any form.
6. Do not name variables after REPL helpers: `var pageInfo = await pageInfo()` breaks `pageInfo()` for the rest of the session. Use names like `info` or `snap`.
7. Act with a tool call every turn. Do not announce what you will do next; call the tool. Reply without a tool call only when you have the final answer.

Example for "find the sign in page on example.com and list its input fields":

Call 1:
var snap = await accessibilitySnapshot();
repl.write(snap.nodes.filter(n => /sign in|log in/i.test(n.name)).map(n => ({role: n.role, name: n.name, backendNodeId: n.backendNodeId})));
Output: [ { role: 'link', name: 'Sign in', backendNodeId: 812 } ]

Call 2:
await click({backendNodeId: 812});
await waitForLoad();
var snap = await accessibilitySnapshot();
var fields = [];
for (const n of snap.nodes.filter(n => ["textbox", "combobox", "checkbox"].includes(n.role))) {
  var visible = await waitForElement({backendNodeId: n.backendNodeId}, {state: "visible", timeoutSec: 1});
  fields.push({role: n.role, name: n.name, visible});
}
repl.write({url: (await pageInfo()).url, fields});
Output: { url: 'https://example.com/login', fields: [ { role: 'textbox', name: 'Email address', visible: true }, { role: 'textbox', name: 'Password', visible: false } ] }

Final answer (no tool call):
- Email address (email)
The password field appears after the email step."""

EVALUATION_CRITERIA = """1. The agent must have navigated to an authentication page (login, sign-up, register, create account).
2. The agent must have identified input fields that are actually visible on the final page.
3. Many sites use progressive disclosure (showing email first, then password on the next step) - this is valid and should be considered successful.
4. The fields in the agent's final answer should match what is visible in the screenshots.
5. Do not penalize for "missing" fields that would only appear in later steps of a multi-step auth flow.
6. If the task asks for "first" input fields, only the initially visible fields need to be reported.
7. The agent should not fill in or submit any forms - just identify the fields."""

AGENT_AUTH = EnvSpec(
    name="agent_auth",
    system_prompt=SYSTEM_PROMPT,
    evaluation_criteria=EVALUATION_CRITERIA,
    task_file="examples/agent_auth/tasks.jsonl",
)
