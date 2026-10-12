"""Offline tests for reporters/slack.py. No network: the opener is injected."""
import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

from reporters import slack

GOOD = "https://hooks.slack.com/services/T000/B000/SECRETSECRET"


class FakeResp:
    def __init__(self, status=200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_opener(status=200, raises=None):
    calls = []

    def opener(req, timeout=None):
        calls.append(req)
        if raises:
            raise raises
        return FakeResp(status)

    opener.calls = calls
    return opener


class TestValidate:
    def test_accepts_slack_webhook(self):
        slack.validate_webhook_url(GOOD)

    @pytest.mark.parametrize("url", [
        "",
        "http://hooks.slack.com/services/T/B/x",               # not https
        "https://example.com/services/T/B/x",                  # other host
        "https://hooks.slack.com.evil.com/services/T/B/x",     # lookalike host
        "https://hooks.slack.com@evil.com/services/T/B/x",     # userinfo trick
        "https://user:pw@hooks.slack.com/services/T/B/x",      # credentials in URL
        "https://hooks.slack.com/other/T/B/x",                 # wrong path
    ])
    def test_rejects(self, url):
        with pytest.raises(ValueError):
            slack.validate_webhook_url(url)


class TestResolve:
    def test_explicit_beats_env(self, monkeypatch):
        monkeypatch.setenv(slack.ENV_VAR, "from-env")
        assert slack.resolve_webhook("explicit") == "explicit"

    def test_env_used_when_no_explicit(self, monkeypatch):
        monkeypatch.setenv(slack.ENV_VAR, f"  {GOOD}\n")
        assert slack.resolve_webhook() == GOOD

    def test_none_when_unset(self, monkeypatch):
        monkeypatch.delenv(slack.ENV_VAR, raising=False)
        assert slack.resolve_webhook() is None


class TestPost:
    def test_success_sends_json_post(self):
        op = make_opener(200)
        r = slack.post("hello", GOOD, opener=op)
        assert r.ok and r.status == 200
        req = op.calls[0]
        assert req.get_method() == "POST"
        assert req.get_header("Content-type") == "application/json"
        assert json.loads(req.data) == {"text": "hello"}

    def test_no_webhook_makes_no_request(self, monkeypatch):
        monkeypatch.delenv(slack.ENV_VAR, raising=False)
        op = make_opener()
        r = slack.post("hello", opener=op)
        assert not r.ok and op.calls == []

    def test_invalid_webhook_makes_no_request(self):
        op = make_opener()
        r = slack.post("hello", "https://evil.example/services/x", opener=op)
        assert not r.ok and op.calls == []

    def test_long_text_truncated(self):
        op = make_opener()
        slack.post("x" * 10000, GOOD, opener=op)
        sent = json.loads(op.calls[0].data)["text"]
        assert len(sent) <= slack.MAX_TEXT and sent.endswith("(truncated)")

    def test_http_error_reports_status(self):
        err = urllib.error.HTTPError(GOOD, 404, "Not Found", {}, io.BytesIO(b""))
        r = slack.post("hello", GOOD, opener=make_opener(raises=err))
        assert not r.ok and r.status == 404

    def test_non_200_is_not_ok(self):
        r = slack.post("hello", GOOD, opener=make_opener(status=500))
        assert not r.ok and r.status == 500

    def test_network_error_never_raises_and_hides_url(self):
        boom = urllib.error.URLError(f"failed to reach {GOOD}")
        r = slack.post("hello", GOOD, opener=make_opener(raises=boom))
        assert not r.ok
        assert "SECRETSECRET" not in (r.error or "")

    def test_timeout_never_raises(self):
        r = slack.post("hello", GOOD, opener=make_opener(raises=TimeoutError("slow")))
        assert not r.ok and "TimeoutError" in r.error


class TestFormat:
    def report(self, n=2, desc="plain"):
        regs = [SimpleNamespace(case_id=f"c{i}", description=desc) for i in range(n)]
        return SimpleNamespace(provider="openai", baseline_timestamp="b", current_timestamp="c", regressions=regs)

    def test_escape(self):
        assert slack.escape_mrkdwn("a & <b> c") == "a &amp; &lt;b&gt; c"

    def test_alert_lists_cases_and_count(self):
        text = slack.format_drift_alert(self.report(2))
        assert "*2 regression(s):*" in text and "`c0`" in text and "`c1`" in text

    def test_alert_escapes_description(self):
        text = slack.format_drift_alert(self.report(1, desc="<!channel> & more"))
        assert "<!channel>" not in text and "&lt;!channel&gt;" in text

    def test_alert_caps_listed_cases(self):
        text = slack.format_drift_alert(self.report(25))
        assert "*25 regression(s):*" in text
        assert text.count("• `c") == slack.MAX_LISTED
        assert "...and 15 more" in text
