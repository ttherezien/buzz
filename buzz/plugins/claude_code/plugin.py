"""
Claude Code plugin for Buzz.

After a transcription is saved, runs the Claude Code CLI installed on the user's
computer in non-interactive mode (``claude -p``) to translate the segments
and/or summarize the transcript. The CLI uses the account the user signed in
with (``claude`` then ``/login``), so no API key is configured in Buzz.

``ANTHROPIC_API_KEY`` is removed from the CLI's environment so it always uses
the signed-in account rather than billing an API key that happens to be set.
All tools are disabled for these calls: Claude only answers with text and never
reads or edits files on the computer.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile

from buzz.plugins.base import (
    BuzzPlugin,
    ConfigField,
    ConfigFieldType,
    PluginContext,
    PluginMetadata,
    plugin_gettext,
)

logger = logging.getLogger(__name__)

_ = plugin_gettext(__file__)

# Segments sent per CLI call when translating, to keep each request well within
# the model's context and output limits.
BATCH_SIZE = 100
TIMEOUT_SECONDS = 600

TRANSLATION_SYSTEM_PROMPT = (
    "You translate transcript segments. The user message is a JSON array of "
    'objects {"i": number, "text": string}. Translate each text into {language}. '
    "Keep the meaning, tone and any names; do not merge or split segments. "
    'Reply with ONLY a JSON array of {"i": number, "text": string} with the same '
    "i values, and no other text or code fences."
)

DEFAULT_SUMMARY_PROMPT = (
    "Summarise the following transcript into a concise set of key points and "
    "action items. Use clear, plain language and write in the language of the "
    "transcript."
)


class ClaudeCodeError(Exception):
    """Raised when the Claude Code CLI cannot be run or returns an error."""


class ClaudeCodePlugin(BuzzPlugin):
    metadata = PluginMetadata(
        id="claude_code",
        name=_("Claude Code"),
        description=_(
            "Translate and/or summarize the transcript with the Claude Code CLI "
            "installed on this computer, using your Claude account instead of an "
            "API key."
        ),
        version="1.0.0",
        pip_dependencies=[],
        config_fields=[
            ConfigField(
                key="claude_path",
                label=_("Path to the claude executable"),
                type=ConfigFieldType.TEXT,
                default="",
                placeholder="claude",
                description=_("Leave empty to find it automatically."),
            ),
            ConfigField(
                key="model",
                label=_("Model"),
                type=ConfigFieldType.TEXT,
                default="",
                placeholder="sonnet",
                description=_(
                    "Leave empty to use the Claude Code default model "
                    "(e.g. sonnet, opus, haiku)."
                ),
            ),
            ConfigField(
                key="translate",
                label=_("Translate segments"),
                type=ConfigFieldType.BOOL,
                default=True,
            ),
            ConfigField(
                key="target_language",
                label=_("Target language"),
                type=ConfigFieldType.TEXT,
                default="English",
            ),
            ConfigField(
                key="summarize",
                label=_("Save a summary to Notes"),
                type=ConfigFieldType.BOOL,
                default=False,
            ),
            ConfigField(
                key="summary_prompt",
                label=_("Summarization prompt"),
                type=ConfigFieldType.TEXTAREA,
                default=DEFAULT_SUMMARY_PROMPT,
            ),
        ],
    )

    def on_complete(self, transcription_id, task, segments, context: PluginContext):
        cfg = context.config
        translate = _coerce_bool(cfg.get("translate", True))
        summarize = _coerce_bool(cfg.get("summarize", False))
        if not translate and not summarize:
            return

        executable = find_claude_executable(cfg.get("claude_path"))
        if not executable:
            context.log.error(
                "Claude Code CLI not found; install it or set its path in the "
                "plugin settings"
            )
            return

        # Read the saved segments rather than the hook's list: they carry the
        # database ids needed to store translations, and reflect any changes
        # made by plugins that ran earlier in on_complete (e.g. resizing).
        rows = context.transcription_service.get_transcription_segments(
            transcription_id=transcription_id
        )
        rows = sorted(rows, key=lambda row: (row.start_time, row.id))
        if not any(row.text.strip() for row in rows):
            context.log.info("Claude Code skipped: empty transcript")
            return

        try:
            if translate:
                self._translate(executable, cfg, rows, context)
            if summarize:
                self._summarize(executable, cfg, transcription_id, rows, context)
        except ClaudeCodeError as exc:
            context.log.error("Claude Code failed: %s", exc)

    def _translate(self, executable, cfg, rows, context):
        language = (cfg.get("target_language") or "").strip() or "English"
        system_prompt = TRANSLATION_SYSTEM_PROMPT.replace("{language}", language)
        model = (cfg.get("model") or "").strip()

        rows = [row for row in rows if row.text.strip()]
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start : start + BATCH_SIZE]
            payload = [{"i": i, "text": row.text} for i, row in enumerate(batch)]
            answer = run_claude(
                executable,
                json.dumps(payload, ensure_ascii=False),
                system_prompt,
                model,
            )
            translations = parse_translations(answer)
            missing = 0
            for i, row in enumerate(batch):
                text = translations.get(i)
                if text:
                    context.transcription_service.update_segment_translation(
                        row.id, text
                    )
                else:
                    missing += 1
            if missing:
                context.log.warning("Claude Code left %d segment(s) untranslated", missing)

    def _summarize(self, executable, cfg, transcription_id, rows, context):
        transcript = "\n".join(row.text.strip() for row in rows if row.text.strip())
        prompt = (cfg.get("summary_prompt") or "").strip() or DEFAULT_SUMMARY_PROMPT
        model = (cfg.get("model") or "").strip()

        summary = run_claude(executable, transcript, prompt, model).strip()
        if summary:
            context.transcription_service.update_transcription_notes(
                transcription_id, summary
            )


def find_claude_executable(configured_path=None):
    """Return the path to the ``claude`` CLI, or ``None`` if it is not found.

    GUI apps (notably on macOS) do not inherit the shell's ``PATH``, so the usual
    install locations are checked when ``claude`` is not on ``PATH``.
    """
    configured_path = (configured_path or "").strip()
    if configured_path:
        return configured_path if os.path.isfile(configured_path) else None

    found = shutil.which("claude")
    if found:
        return found

    home = os.path.expanduser("~")
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", os.path.join(home, "AppData", "Roaming"))
        candidates = [
            os.path.join(home, ".local", "bin", "claude.exe"),
            os.path.join(appdata, "npm", "claude.cmd"),
        ]
    else:
        candidates = [
            os.path.join(home, ".local", "bin", "claude"),
            os.path.join(home, ".claude", "local", "claude"),
            "/opt/homebrew/bin/claude",
            "/usr/local/bin/claude",
        ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def run_claude(executable, prompt, system_prompt, model=""):
    """Run ``claude -p`` with tools disabled and return the text of its answer.

    The prompt goes through stdin to avoid command-line length limits.
    """
    cmd = [
        executable,
        "-p",
        "--output-format",
        "json",
        "--tools",
        "",
        "--no-session-persistence",
        "--system-prompt",
        system_prompt,
    ]
    if model:
        cmd += ["--model", model]

    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    # An empty working directory keeps the CLI from loading a project's
    # CLAUDE.md or settings from wherever Buzz happens to run.
    with tempfile.TemporaryDirectory() as cwd:
        try:
            completed = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=cwd,
                env=_cli_env(),
                timeout=TIMEOUT_SECONDS,
                **kwargs,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ClaudeCodeError(str(exc)) from exc

    try:
        output = json.loads(completed.stdout)
    except json.JSONDecodeError:
        detail = (completed.stderr or completed.stdout).strip()
        raise ClaudeCodeError(
            f"unexpected output (exit code {completed.returncode}): {detail[:500]}"
        )

    if completed.returncode != 0 or output.get("is_error"):
        raise ClaudeCodeError(str(output.get("result") or completed.stderr)[:500])
    return output.get("result") or ""


def parse_translations(answer):
    """Map segment index to translated text from Claude's JSON answer.

    Tolerates code fences or prose around the array; returns ``{}`` when no
    array can be read.
    """
    start, end = answer.find("["), answer.rfind("]")
    if start == -1 or end <= start:
        return {}
    try:
        items = json.loads(answer[start : end + 1])
    except json.JSONDecodeError:
        return {}

    translations = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        index, text = item.get("i"), item.get("text")
        if isinstance(index, int) and isinstance(text, str) and text.strip():
            translations[index] = text.strip()
    return translations


def _cli_env():
    env = dict(os.environ)
    # Use the signed-in Claude account, never an API key.
    env.pop("ANTHROPIC_API_KEY", None)
    # PyInstaller points LD_LIBRARY_PATH at Buzz's bundled libraries; restore the
    # original so the CLI does not load them.
    if getattr(sys, "frozen", False):
        original = env.pop("LD_LIBRARY_PATH_ORIG", None)
        if original is not None:
            env["LD_LIBRARY_PATH"] = original
        else:
            env.pop("LD_LIBRARY_PATH", None)
    return env


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    if isinstance(value, int):
        return value != 0
    return bool(value)
