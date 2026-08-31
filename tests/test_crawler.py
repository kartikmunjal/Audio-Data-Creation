import io
import tarfile
from pathlib import Path

import pytest

from audio_curation.crawler import (
    CrawlPolicy, PoliteSession, RobotsDenied, count_audio_members,
    extract_librispeech_archive, parse_openslr_page,
)


PAGE = """
<p><b>Identifier:</b> SLR31</p>
<p><b>Summary:</b> Regression corpus</p>
<p><b>Category:</b> Speech</p>
<p><b>License:</b> CC BY 4.0</p>
<a href="https://mirror.test/resources/31/dev-clean-2.tar.gz">dev-clean-2.tar.gz</a>
<a href="https://other.test/resources/31/dev-clean-2.tar.gz">mirror</a>
<a href="/resources/31/train-clean-5.tar.gz">train-clean-5.tar.gz</a>
"""


def test_parse_openslr_page_deduplicates_mirrors():
    parsed = parse_openslr_page(PAGE, "https://www.openslr.org/31/")
    assert parsed["identifier"] == "SLR31"
    assert parsed["license"] == "CC BY 4.0"
    assert [item["filename"] for item in parsed["archives"]] == [
        "dev-clean-2.tar.gz", "train-clean-5.tar.gz"
    ]
    assert len(parsed["archives"][0]["mirrors"]) == 2


class Response:
    def __init__(self, text: str):
        self.text = text
    def raise_for_status(self):
        return None


def test_polite_session_honors_declared_delay(monkeypatch):
    responses = iter([Response("User-agent: *\nDisallow:\nCrawl-delay: 7\n"), Response("ok")])
    waits = []
    client = PoliteSession(CrawlPolicy(min_delay_sec=5), sleep=waits.append)
    monkeypatch.setattr(client.session, "get", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr("audio_curation.crawler.time.monotonic", lambda: 1.0)
    client.get("https://example.test/allowed")
    assert waits == [6.0]
    evidence = client.robots_evidence()["https://example.test"]
    assert evidence["effective_delay_sec"] == 7.0


def test_polite_session_rejects_disallowed_url(monkeypatch):
    response = Response("User-agent: *\nDisallow: /private\n")
    client = PoliteSession(CrawlPolicy(), sleep=lambda _: None)
    monkeypatch.setattr(client.session, "get", lambda *args, **kwargs: response)
    with pytest.raises(RobotsDenied):
        client.get("https://example.test/private/file")


def _add(archive, name: str, payload: bytes):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def test_extract_manifest_contract_and_cap(tmp_path: Path):
    archive_path = tmp_path / "sample.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        _add(archive, "LibriSpeech/dev/1/2/1-2.trans.txt", b"1-2-3 FIRST SENTENCE\n1-2-4 SECOND SENTENCE\n")
        _add(archive, "LibriSpeech/dev/1/2/1-2-4.flac", b"flac-two")
        _add(archive, "LibriSpeech/dev/1/2/1-2-3.flac", b"flac-one")
    source = {
        "identifier": "SLR31", "page_url": "https://www.openslr.org/31/",
        "archive_url": "https://www.openslr.org/resources/31/sample.tar.gz",
        "license": "CC BY 4.0",
    }
    frame = extract_librispeech_archive(
        archive_path, tmp_path / "audio", max_clips=1, source=source
    )
    assert count_audio_members(archive_path) == 2
    assert frame[["id", "speaker_id", "sentence"]].iloc[0].to_dict() == {
        "id": "1-2-3", "speaker_id": "1", "sentence": "FIRST SENTENCE"
    }
    assert frame.iloc[0]["source_license"] == "CC BY 4.0"
    assert Path(frame.iloc[0]["path"]).read_bytes() == b"flac-one"


def test_archive_path_traversal_is_not_extracted(tmp_path: Path):
    archive_path = tmp_path / "sample.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        _add(archive, "../1-2.trans.txt", b"1-2-3 BAD\n")
        _add(archive, "../1-2-3.flac", b"bad")
    source = {"identifier": "SLR31", "page_url": "p", "archive_url": "a", "license": "CC BY 4.0"}
    frame = extract_librispeech_archive(archive_path, tmp_path / "audio", max_clips=None, source=source)
    assert frame.empty
    assert not (tmp_path / "1-2-3.flac").exists()
