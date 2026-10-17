import json
import logging
import os
import subprocess

import pytest

from buzz.plugins.base import PluginContext
from buzz.plugins.claude_code import plugin as cc


class _DBSeg:
    def __init__(self, id, text, start):
        self.id = id
        self.text = text
        self.start_time = start
        self.end_time = start + 1000


class _Service:
    def __init__(self, rows):
        self.rows = rows
        self.translations = {}
        self.notes = None

    def get_transcription_segments(self, transcription_id):
        return self.rows

    def update_segment_translation(self, segment_id, translation):
        self.translations[segment_id] = translation

    def update_transcription_notes(self, transcription_id, notes):
        self.notes = (transcription_id, notes)


class _FakeCli:
    """Stands in for ``subprocess.run`` and answers like ``claude -p``."""

    def __init__(self, answer=None, is_error=False, returncode=0):
        self.answer = answer
        self.is_error = is_error
        self.returncode = returncode
        self.calls = []

    def __call__(self, cmd, input, env, **kwargs):
        self.calls.append({"cmd": cmd, "input": input, "env": env})
        if self.answer is not None:
            result = self.answer
        else:
            # Echo a translation for every segment sent.
            items = json.loads(input)
            result = json.dumps([{"i": it["i"], "text": "T:" + it["text"]} for it in items])
        stdout = json.dumps({"type": "result", "is_error": self.is_error, "result": result})
        return subprocess.CompletedProcess(cmd, self.returncode, stdout=stdout, stderr="")


def _context(service, **config):
    cfg = {
        "claude_path": "",
        "model": "",
        "translate": True,
        "target_language": "French",
        "summarize": False,
        "summary_prompt": "",
    }
    cfg.update(config)
    return PluginContext(
        config=cfg,
        transcription_service=service,
        settings=None,
        logger=logging.getLogger("test"),
    )


@pytest.fixture()
def fake_cli(monkeypatch):
    def install(**kwargs):
        cli = _FakeCli(**kwargs)
        monkeypatch.setattr(cc.subprocess, "run", cli)
        monkeypatch.setattr(cc, "find_claude_executable", lambda path: "/usr/bin/claude")
        return cli

    return install


def test_claude_code_loads():
    from buzz.plugins.loader import load_plugin_from_dir

    plugin = load_plugin_from_dir("buzz/plugins/claude_code")
    assert plugin.metadata.id == "claude_code"
    assert plugin.metadata.pip_dependencies == []


def test_claude_code_locales_cover_all_strings():
    locale_dir = "buzz/plugins/claude_code/locale"
    expected = set(os.listdir("buzz/plugins/ai_summary/locale"))
    assert set(os.listdir(locale_dir)) == expected

    keys = None
    for name in expected:
        with open(os.path.join(locale_dir, name), encoding="utf-8") as f:
            translations = json.load(f)
        assert all(translations.values()), name
        keys = keys or set(translations)
        assert set(translations) == keys, name


def test_translates_segments_into_database(fake_cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-used")
    cli = fake_cli()
    service = _Service([_DBSeg(2, "World", 1000), _DBSeg(1, "Hello", 0)])

    cc.ClaudeCodePlugin().on_complete("tid", None, [], _context(service))

    assert service.translations == {1: "T:Hello", 2: "T:World"}
    assert service.notes is None

    call = cli.calls[0]
    assert call["cmd"][:2] == ["/usr/bin/claude", "-p"]
    # Tools are disabled and the prompt goes through stdin.
    assert call["cmd"][call["cmd"].index("--tools") + 1] == ""
    assert "French" in call["cmd"][call["cmd"].index("--system-prompt") + 1]
    assert json.loads(call["input"]) == [{"i": 0, "text": "Hello"}, {"i": 1, "text": "World"}]
    assert "ANTHROPIC_API_KEY" not in call["env"]


def test_translates_in_batches(fake_cli):
    cli = fake_cli()
    rows = [_DBSeg(i, f"s{i}", i * 1000) for i in range(cc.BATCH_SIZE + 5)]
    service = _Service(rows)

    cc.ClaudeCodePlugin().on_complete("tid", None, [], _context(service))

    assert len(cli.calls) == 2
    assert len(service.translations) == len(rows)
    assert service.translations[cc.BATCH_SIZE + 4] == f"T:s{cc.BATCH_SIZE + 4}"


def test_summary_goes_to_notes(fake_cli):
    cli = fake_cli(answer="Key points.")
    service = _Service([_DBSeg(1, "Hello", 0)])

    cc.ClaudeCodePlugin().on_complete(
        "tid",
        None,
        [],
        _context(service, translate=False, summarize=True, model="haiku"),
    )

    assert service.notes == ("tid", "Key points.")
    assert cli.calls[0]["cmd"][-2:] == ["--model", "haiku"]


def test_cli_error_leaves_transcript_unchanged(fake_cli):
    fake_cli(answer="Not logged in", is_error=True, returncode=1)
    service = _Service([_DBSeg(1, "Hello", 0)])

    cc.ClaudeCodePlugin().on_complete("tid", None, [], _context(service, summarize=True))

    assert service.translations == {}
    assert service.notes is None


def test_missing_cli_does_nothing(monkeypatch):
    monkeypatch.setattr(cc, "find_claude_executable", lambda path: None)
    service = _Service([_DBSeg(1, "Hello", 0)])

    cc.ClaudeCodePlugin().on_complete("tid", None, [], _context(service))

    assert service.translations == {}


def test_parse_translations_tolerates_wrapping():
    answer = 'Here you go:\n```json\n[{"i": 0, "text": "Bonjour"}, {"i": "x"}]\n```'
    assert cc.parse_translations(answer) == {0: "Bonjour"}
    assert cc.parse_translations("no json here") == {}
    assert cc.parse_translations("[not json]") == {}


def test_configured_path_must_exist(tmp_path):
    exe = tmp_path / "claude"
    exe.write_text("")
    assert cc.find_claude_executable(str(exe)) == str(exe)
    assert cc.find_claude_executable(str(tmp_path / "missing")) is None
