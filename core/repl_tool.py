"""
The agent's single tool: JavaScript evaluation in the Kernel Browser REPL.

The REPL is a persistent Node.js runtime inside the browser VM with browser-control
helpers (gotoUrl, click, fillInput, accessibilitySnapshot, js, cdp, ...) preloaded as
globals. See https://github.com/kernel/kernel-images/blob/main/server/docs/repl.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel import APIError, APIStatusError, AsyncKernel
from kernel.types import BrowserReplResult
from tinker_cookbook.renderers.base import ToolSpec
from tinker_cookbook.tool_use import ToolInput, ToolResult, error_tool_result, simple_tool_result

MAX_OUTPUT_CHARS = 4000

# Pooled browsers come back with the previous occupant's tabs, cookies, and storage, so
# each episode runs in a fresh browser context and disposes every older one. The default
# context cannot be disposed (REPL helpers like newTab() open tabs there), so clear its
# cookies and close its tabs; the episode's window keeps the browser open.
_SETUP_JS = """
var fresh = (await cdp("Target.createBrowserContext", {}, null)).browserContextId;
var {targetId: episodeTab} = await cdp("Target.createTarget", {url: "about:blank", browserContextId: fresh}, null);
for (const id of (await cdp("Target.getBrowserContexts", {}, null)).browserContextIds) {
  if (id !== fresh) await cdp("Target.disposeBrowserContext", {browserContextId: id}, null);
}
await cdp("Storage.clearCookies", {}, null);
var {targetInfos} = await cdp("Target.getTargets", {}, null);
for (const t of targetInfos.filter(t => t.type === "page" && t.browserContextId !== fresh)) {
  await cdp("Target.closeTarget", {targetId: t.targetId}, null);
}
await switchTab(episodeTab);
await gotoUrl(%s);
await waitForLoad();
"""

_REFERENCE_PATH = Path(__file__).parent / "repl_reference.md"
REPL_REFERENCE = re.sub(r"<!--.*?-->", "", _REFERENCE_PATH.read_text(), flags=re.S).strip()

BROWSER_REPL_DESCRIPTION = f"""Execute JavaScript in a persistent Node.js REPL attached to the browser.

- Nothing is returned unless you print it. Expression values are discarded; end each call with `repl.write(value)` or `console.log(value)`.
- The REPL is Node.js, not the page: `document`, `window`, and other page globals exist only inside `js(() => ...)`. A `js()` function cannot see REPL variables, so pass data with `js(fn, {{arg}})`.
- The browser helpers below are globals, e.g. `await gotoUrl(url)` or `await click(target)`. Top-level `await` works; top-level `return` does not.
- Top-level bindings persist between calls. Declare them with `var`: a `const` or `let` name cannot be declared again in a later call.
- Output longer than {MAX_OUTPUT_CHARS} characters is truncated. Filter large results such as `accessibilitySnapshot()` nodes before printing them.
- Exceptions are returned as errors and do not clear REPL state.

{REPL_REFERENCE}"""


def format_repl_result(result: BrowserReplResult, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Render a REPL result as the text the model sees."""
    parts: list[str] = []
    for item in result.content or []:
        if item.type == "text":
            parts.append(item.text)
        else:
            parts.append("[image output omitted]")
    if not result.success:
        parts.append(f"\nError: {result.error}")
    if result.repl_terminated:
        parts.append("\nThe REPL process was terminated; the next call starts with empty state.")

    text = "".join(parts).strip() or "(no output)"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated {len(text) - max_chars} characters]"
    return text


@dataclass
class ReplCall:
    """One executed REPL cell, kept for logging and reward computation."""

    code: str
    output: str
    failed: bool
    screenshot: bytes | None = None


class BrowserReplTool:
    """Implements the tinker_cookbook Tool protocol on top of one pooled Kernel browser."""

    name = "browser_repl"
    description = BROWSER_REPL_DESCRIPTION
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "JavaScript to evaluate. Print results with repl.write(...).",
            }
        },
        "required": ["code"],
    }

    def __init__(
        self,
        kernel: AsyncKernel,
        pool_name: str,
        initial_url: str,
        acquire_timeout_seconds: int = 600,
        timeout_sec: int = 30,
    ):
        self.kernel = kernel
        self.pool_name = pool_name
        self.initial_url = initial_url
        self.acquire_timeout_seconds = acquire_timeout_seconds
        self.timeout_sec = timeout_sec
        self.session_id: str | None = None
        self.calls: list[ReplCall] = []
        self.reusable = True

    def to_spec(self) -> ToolSpec:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    async def run(self, input: ToolInput) -> ToolResult:
        call_id = input.call_id or ""
        code = input.arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return error_tool_result(
                "`code` must be a non-empty string", call_id=call_id, name=self.name
            )

        session_id = await self._ensure_session()
        try:
            result = await self.kernel.browsers.repl(
                session_id, code=code, timeout_sec=self.timeout_sec
            )
        except APIStatusError as e:
            # 404/410 mean the browser is gone; nothing more can happen in this episode.
            session_lost = e.status_code in (404, 410)
            if session_lost:
                self.reusable = False
            self.calls.append(ReplCall(code=code, output=f"Error: {e.message}", failed=True))
            return error_tool_result(
                f"Browser REPL request failed: {e.message}",
                call_id=call_id,
                name=self.name,
                should_stop=session_lost,
            )

        output = format_repl_result(result)
        self.calls.append(
            ReplCall(
                code=code,
                output=output,
                failed=not result.success,
                screenshot=await self.screenshot(),
            )
        )
        return simple_tool_result(output, call_id=call_id, name=self.name)

    async def _ensure_session(self) -> str:
        """Acquire a pooled browser and open the start URL in a clean context on first use.

        Lazy so that envs which never run (e.g. spares created by a retry) hold no browser.
        """
        if self.session_id is None:
            browser = await self.kernel.browser_pools.acquire(
                self.pool_name, acquire_timeout_seconds=self.acquire_timeout_seconds
            )
            self.session_id = browser.session_id
            try:
                result = await self.kernel.browsers.repl(
                    self.session_id,
                    code=_SETUP_JS % json.dumps(self.initial_url),
                    reset=True,
                    timeout_sec=self.timeout_sec,
                )
                if not result.success:
                    raise RuntimeError(f"Failed to load {self.initial_url}: {result.error}")
            except BaseException:
                # The browser's state is unknown; have the pool replace it.
                self.reusable = False
                raise
        return self.session_id

    async def release(self) -> None:
        """Return the browser to the pool, or have the pool replace it if it is unusable."""
        if self.session_id is None:
            return
        session_id, self.session_id = self.session_id, None
        await self.kernel.browser_pools.release(
            self.pool_name, session_id=session_id, reuse=self.reusable
        )

    async def screenshot(self) -> bytes | None:
        if self.session_id is None:
            return None
        try:
            response = await self.kernel.browsers.computer.capture_screenshot(self.session_id)
            return await response.read()
        except APIError:
            return None
