"""Tests for the PreToolUse check in .claude/hooks/_check.py."""

import importlib.util
from pathlib import Path

import pytest

HOOK_CHECK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "_check.py"


@pytest.fixture
def hook():
    """Load _check.py as a module (it lives outside src/)."""
    spec = importlib.util.spec_from_file_location("hook_check", HOOK_CHECK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "cmd",
    [
        "uv sync",
        "uv sync --extra gpu",
        "cd notebooks && uv run python -m pytest",
        "uvx ruff check",
        "pip install lightgbm",
        "pip3 install lightgbm",
        "pip uninstall -y pandas",
        "python -m pip install lightgbm",
        "py -3.13 -m pip install lightgbm",
        r"C:\Users\abc\Documents\WPy64-313150\python\python.exe -m pip install lightgbm",
        "/c/Users/abc/Documents/WPy64-313150/python/python.exe -m pip install lightgbm",
        r'"C:\Program Files\Python313\python.exe" -m pip install lightgbm',
        "PIP_INDEX_URL=https://example.org/simple pip install lightgbm",
        "echo start; pip install lightgbm",
        "pipx install ruff",
        "conda install lightgbm",
    ],
)
def test_install_commands_are_detected(hook, cmd):
    """Package installs are found as a command of their own, also with path or env vars."""
    assert hook.INSTALL_CMD.search(cmd)


@pytest.mark.parametrize(
    "cmd",
    [
        "python -m pytest",
        "python -m ruff check --fix src tests",
        "python -m pip list",
        "pip show lightgbm",
        "uv --version",
        "echo uv sync",
        'git commit -m "Document uv sync and pip install for the uv setup"',
        'grep -rn "pip install" README.md',
    ],
)
def test_other_commands_are_not_installs(hook, cmd):
    """Mentioning an install command in text or reading package info is not an install."""
    assert not hook.INSTALL_CMD.search(cmd)


def test_installs_blocked_only_without_venv(hook, tmp_path, monkeypatch):
    """Without .venv (WinPython setup) installs are blocked; the uv setup with .venv keeps them."""
    monkeypatch.setattr(hook, "PROJECT_DIR", tmp_path)
    assert hook.check("Bash", {"command": "pip install lightgbm"})
    assert hook.check("Bash", {"command": "uv sync"})
    assert hook.check("Bash", {"command": "python -m pytest"}) is None

    (tmp_path / ".venv").mkdir()
    assert hook.check("Bash", {"command": "pip install lightgbm"}) is None
    assert hook.check("Bash", {"command": "uv sync"}) is None


def test_existing_checks_still_apply(hook, tmp_path, monkeypatch):
    """Credential and SQL checks do not depend on the setup."""
    monkeypatch.setattr(hook, "PROJECT_DIR", tmp_path)
    (tmp_path / ".venv").mkdir()
    assert hook.check("Edit", {"file_path": "config/.env"})
    assert hook.check("Bash", {"command": "snowsql -q 'DROP TABLE x'"})
    assert hook.check("Edit", {"file_path": "notebooks/work_ruestmengen.ipynb"}) is None
