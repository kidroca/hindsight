"""Load the directory plugin against actual Hermes; run apart from stubbed tests/."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "hindsight_hermes", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
assert spec and spec.loader
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(plugin, "_warn_if_client_outdated", lambda: None)
    monkeypatch.setattr(plugin, "_check_api_supports_update_mode_append", lambda *a, **k: True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Integration tests must never create a network client or daemon")

    monkeypatch.setattr(plugin.HindsightMemoryProvider, "_new_cloud_client", forbidden)
    monkeypatch.setattr(plugin.HindsightMemoryProvider, "_start_embedded_daemon", forbidden)
