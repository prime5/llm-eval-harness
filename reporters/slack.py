"""Post short messages to Slack through an incoming webhook.

Reporting must never break an eval run, so `post` returns a Result instead of
raising, and no error text ever contains the webhook URL (it is a credential:
anyone holding it can post to the channel).

Prefer the SLACK_WEBHOOK_URL environment variable over a command-line flag: a
URL on the command line lands in shell history and CI logs.

Check a webhook from a terminal without printing it:

    read -s "SLACK_WEBHOOK_URL?Webhook URL (hidden): "; export SLACK_WEBHOOK_URL
    python3 -m reporters.slack "harness test message"
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlparse

ENV_VAR = "SLACK_WEBHOOK_URL"
MAX_TEXT = 3500          # keep messages short; longer text is truncated with a note
MAX_LISTED = 10          # cases listed individually in a drift alert
_ALLOWED_HOST = "hooks.slack.com"


@dataclass
class Result:
    ok: bool
    status: Optional[int] = None
    error: Optional[str] = None


def validate_webhook_url(url: str) -> None:
    """Raise ValueError unless url is an https Slack incoming-webhook URL.

    Refusing other hosts stops a mistyped or poisoned value from sending
    message text (which may carry eval details) to an arbitrary server.
    """
    parsed = urlparse(url or "")
    if parsed.scheme != "https":
        raise ValueError("webhook URL must use https")
    if parsed.hostname != _ALLOWED_HOST or parsed.username or parsed.password:
        raise ValueError(f"webhook URL host must be {_ALLOWED_HOST}")
    if not parsed.path.startswith("/services/"):
        raise ValueError("webhook URL path must start with /services/")


def resolve_webhook(explicit: Optional[str] = None) -> Optional[str]:
    """The explicit value if given, else the environment variable, else None."""
    value = (explicit or os.environ.get(ENV_VAR) or "").strip()
    return value or None


def escape_mrkdwn(text: str) -> str:
    """Escape the three characters Slack treats as control characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _truncate(text: str) -> str:
    if len(text) <= MAX_TEXT:
        return text
    return text[: MAX_TEXT - 20].rstrip() + "\n...(truncated)"


def post(
    text: str,
    webhook_url: Optional[str] = None,
    timeout: float = 10,
    opener: Callable = urllib.request.urlopen,
) -> Result:
    """Send `text` to Slack. Never raises; never echoes the webhook URL."""
    url = resolve_webhook(webhook_url)
    if url is None:
        return Result(False, None, f"no webhook configured (set {ENV_VAR})")
    try:
        validate_webhook_url(url)
    except ValueError as exc:
        return Result(False, None, str(exc))

    body = json.dumps({"text": _truncate(text)}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with opener(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None)
            return Result(status == 200, status, None if status == 200 else f"HTTP {status}")
    except urllib.error.HTTPError as exc:
        return Result(False, exc.code, f"HTTP {exc.code}")
    except Exception as exc:  # network down, timeout, TLS, ...
        return Result(False, None, _scrub(f"{type(exc).__name__}: {exc}", url))


def _scrub(message: str, url: str) -> str:
    return message.replace(url, "<webhook>")


def format_drift_alert(report) -> str:
    """Slack text for a drift report with regressions.

    Expects the DriftReport shape in evals/drift_detector.py: provider,
    baseline_timestamp, current_timestamp, regressions (items with case_id and
    description).
    """
    regs = list(report.regressions)
    lines = [
        f"• `{escape_mrkdwn(str(c.case_id))}`: {escape_mrkdwn(str(c.description)[:60])}"
        for c in regs[:MAX_LISTED]
    ]
    if len(regs) > MAX_LISTED:
        lines.append(f"...and {len(regs) - MAX_LISTED} more")
    return (
        ":rotating_light: *LLM Eval Regression Detected*\n"
        f"Provider: `{escape_mrkdwn(str(report.provider))}`\n"
        f"Baseline: `{escape_mrkdwn(str(report.baseline_timestamp))}`\n"
        f"Current:  `{escape_mrkdwn(str(report.current_timestamp))}`\n\n"
        f"*{len(regs)} regression(s):*\n" + "\n".join(lines)
    )


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print('usage: python3 -m reporters.slack "message"', file=sys.stderr)
        return 2
    result = post(" ".join(argv))
    if result.ok:
        print("posted")
        return 0
    print(f"not posted: {result.error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
