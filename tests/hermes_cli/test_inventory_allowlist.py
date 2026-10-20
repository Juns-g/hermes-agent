"""Profile model allowlists filter shared picker visibility, not discovery."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from hermes_cli.inventory import build_models_payload, load_picker_context


@pytest.mark.parametrize("restriction", [None, ["selected"], []])
def test_profile_allowlist_filters_only_configured_provider(tmp_path, monkeypatch, restriction):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = "model:\n  provider: openai-codex\n  default: current\n"
    if restriction is not None:
        config += f"model_catalog:\n  allowed_models:\n    openai-codex: {restriction!r}\n"
    (tmp_path / "config.yaml").write_text(config)
    models = ["selected", "hidden", "current"]
    discovered = [
        {"slug": "openai-codex", "models": models, "total_models": len(models),
         "is_current": True},
        {"slug": "gemini", "models": ["other"], "total_models": 1, "is_current": False},
    ]
    with (
        patch("hermes_cli.model_switch.list_authenticated_providers",
              side_effect=lambda **kwargs: deepcopy(discovered)),
        patch("hermes_cli.inventory._local_runtime_row", return_value=None),
    ):
        ctx = load_picker_context()
        payload = build_models_payload(ctx)
        live = build_models_payload(ctx.with_overrides(current_model="hidden"))

    expected = models if restriction is None else [*restriction, "current"]
    assert payload["providers"][0]["models"] == expected
    assert payload["providers"][0]["total_models"] == len(expected)
    assert payload["providers"][1] == discovered[1]
    assert payload["model"] == "current"
    assert payload["provider"] == "openai-codex"
    assert "hidden" in live["providers"][0]["models"]
    assert live["model"] == "hidden"
    assert discovered[0]["models"] == models
