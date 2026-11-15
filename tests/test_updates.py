import hashlib
import io
import json
import urllib.error
from dataclasses import replace

import pytest

from race_engineer import updates

BODY = b"MZ-unit-test-installer-not-executable"


def available(version="0.2.0"):
    filename = f"Pitward-Setup-{version}-windows-x64.exe"
    prefix = f"https://github.com/{updates.REPOSITORY}/releases/download/v{version}/"
    return updates.AvailableUpdate(
        version,
        filename,
        prefix + filename,
        hashlib.sha256(BODY).hexdigest(),
        len(BODY),
        f"https://github.com/{updates.REPOSITORY}/releases/tag/v{version}",
    )


def metadata(release):
    return {
        "draft": False,
        "prerelease": False,
        "tag_name": "v" + release.version,
        "assets": [
            {
                "name": release.filename,
                "browser_download_url": release.url,
                "size": release.size,
                "digest": "sha256:" + release.sha256,
            },
            {
                "name": "SHA256SUMS.txt",
                "browser_download_url": release.url.rsplit("/", 1)[0] + "/SHA256SUMS.txt",
            },
        ],
    }


@pytest.fixture
def release_server(monkeypatch):
    release = available()
    raw = metadata(release)
    calls = []

    def read(url, limit, cancelled):
        calls.append(url)
        if url == updates.LATEST_URL:
            return json.dumps(raw).encode()
        return f"{release.sha256}  {release.filename}\n".encode()

    monkeypatch.setattr(updates, "_read", read)
    return release, raw, calls


def test_latest_release_requires_matching_checksum_and_canonical_installer(release_server):
    release, _, calls = release_server
    assert updates.check_for_update("0.1.0") == release
    assert calls == [updates.LATEST_URL, release.url.rsplit("/", 1)[0] + "/SHA256SUMS.txt"]


@pytest.mark.parametrize("current", ["0.2.0", "0.3.0", "0.10.0"])
def test_no_downgrade_or_same_version_and_no_installer_fetch(release_server, current):
    _, _, calls = release_server
    assert updates.check_for_update(current) is None
    assert calls == [updates.LATEST_URL]


@pytest.mark.parametrize("field", ["draft", "prerelease"])
def test_drafts_and_previews_are_not_offered(release_server, field):
    _, raw, calls = release_server
    raw[field] = True
    assert updates.check_for_update("0.1.0") is None
    assert len(calls) == 1


def test_stable_version_orders_after_its_release_candidate(release_server):
    assert updates.check_for_update("0.2.0rc1") == release_server[0]
    assert updates.version_key("0.10.0") > updates.version_key("0.9.0")


@pytest.mark.parametrize(
    "problem",
    ["external_url", "missing_checksums", "duplicate_asset", "bad_size", "digest_mismatch"],
)
def test_incomplete_or_untrusted_release_assets_are_rejected(release_server, problem):
    _, raw, _ = release_server
    if problem == "external_url":
        raw["assets"][0]["browser_download_url"] = "https://example.com/installer.exe"
    elif problem == "missing_checksums":
        raw["assets"].pop()
    elif problem == "duplicate_asset":
        raw["assets"].append(raw["assets"][0])
    elif problem == "bad_size":
        raw["assets"][0]["size"] = 0
    else:
        raw["assets"][0]["digest"] = "sha256:" + "a" * 64
    with pytest.raises(updates.UpdateError):
        updates.check_for_update("0.1.0")


@pytest.mark.parametrize("status", [403, 429, 500])
def test_network_failures_have_display_safe_codes(monkeypatch, status):
    def fail(*args):
        raise urllib.error.HTTPError(updates.LATEST_URL, status, "private detail", {}, None)

    monkeypatch.setattr(updates, "_read", fail)
    with pytest.raises(updates.UpdateError, match=r"^update_check_unavailable$"):
        updates.check_for_update("0.1.0")


def test_no_published_release_is_not_an_error(monkeypatch):
    def fail(*args):
        raise urllib.error.HTTPError(updates.LATEST_URL, 404, "not found", {}, None)

    monkeypatch.setattr(updates, "_read", fail)
    assert updates.check_for_update("0.1.0") is None


def test_cancelled_check_never_opens_network(monkeypatch):
    monkeypatch.setattr(updates, "_open", lambda *_: pytest.fail("cancelled before request"))
    with pytest.raises(updates.UpdateCancelled):
        updates.check_for_update("0.1.0", cancelled=lambda: True)


def test_download_verifies_before_exposing_an_installer_and_reuses_verified_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(updates, "_open", lambda _: io.BytesIO(BODY))
    progress = []
    result = updates.download_update(available(), tmp_path, progress=progress.append)
    assert result.read_bytes() == BODY and progress[-1] == 100
    assert not list(tmp_path.glob("*.partial"))
    monkeypatch.setattr(updates, "_open", lambda _: pytest.fail("verified cache should be reused"))
    assert updates.download_update(available(), tmp_path) == result


@pytest.mark.parametrize("body", [BODY[:-1], BODY + b"extra", b"x" * len(BODY)])
def test_truncated_extra_or_corrupted_downloads_cannot_be_installed(tmp_path, monkeypatch, body):
    monkeypatch.setattr(updates, "_open", lambda _: io.BytesIO(body))
    with pytest.raises(updates.UpdateError, match="update_checksum_mismatch"):
        updates.download_update(available(), tmp_path)
    assert not (tmp_path / available().filename).exists()


def test_cancelled_download_never_becomes_an_installer(tmp_path, monkeypatch):
    checks = iter([False, True])
    monkeypatch.setattr(updates, "_open", lambda _: io.BytesIO(BODY))
    with pytest.raises(updates.UpdateCancelled):
        updates.download_update(available(), tmp_path, cancelled=lambda: next(checks))
    assert not (tmp_path / available().filename).exists()


def test_cache_paths_cannot_escape_the_owned_directory(tmp_path):
    with pytest.raises(updates.UpdateError, match="update_cache_invalid"):
        updates.download_update(replace(available(), filename="../unexpected.exe"), tmp_path)
    assert not (tmp_path.parent / "unexpected.exe").exists()


def test_installer_handoff_uses_the_existing_directory_and_preserves_custom_paths(tmp_path):
    command = updates.installer_command(
        tmp_path / "update.exe",
        tmp_path / "custom install",
        tmp_path / "custom config.toml",
        tmp_path / "custom settings.json",
    )
    assert command == [
        str(tmp_path / "update.exe"),
        "/NORESTART",
        f"/DIR={tmp_path / 'custom install'}",
        f"/PITWARDCONFIG={tmp_path / 'custom config.toml'}",
        f"/PITWARDSETTINGS={tmp_path / 'custom settings.json'}",
    ]
    assert not any("SILENT" in arg or "unins" in arg for arg in command)
