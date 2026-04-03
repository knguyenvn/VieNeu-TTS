from pathlib import Path

import pytest

from vieneu.turbo import _resolve_model_asset


def test_resolve_model_asset_prefers_existing_file(tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_text("stub", encoding="utf-8")

    resolved = _resolve_model_asset(str(model_path), "ignored.gguf")

    assert resolved == str(model_path)


def test_resolve_model_asset_uses_file_inside_existing_directory(tmp_path):
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    asset_path = model_dir / "model.gguf"
    asset_path.write_text("stub", encoding="utf-8")

    resolved = _resolve_model_asset(str(model_dir), "model.gguf")

    assert resolved == str(asset_path)


def test_resolve_model_asset_falls_back_to_local_cache(monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            return "/tmp/cached-model.gguf"
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("vieneu.turbo.hf_hub_download", fake_download)

    resolved = _resolve_model_asset("repo/id", "model.gguf")

    assert resolved == "/tmp/cached-model.gguf"
    assert calls == [
        {"repo_id": "repo/id", "filename": "model.gguf", "token": None},
        {
            "repo_id": "repo/id",
            "filename": "model.gguf",
            "token": None,
            "local_files_only": True,
        },
    ]


def test_resolve_model_asset_falls_back_to_filename_path(monkeypatch, tmp_path):
    local_asset = tmp_path / "model.gguf"
    local_asset.write_text("stub", encoding="utf-8")

    def fake_download(**kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("vieneu.turbo.hf_hub_download", fake_download)

    resolved = _resolve_model_asset("repo/id", str(local_asset))

    assert resolved == str(local_asset)


def test_resolve_model_asset_raises_when_no_source_found(monkeypatch):
    def fake_download(**kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("vieneu.turbo.hf_hub_download", fake_download)

    with pytest.raises(FileNotFoundError):
        _resolve_model_asset("repo/id", "missing-model.gguf")
